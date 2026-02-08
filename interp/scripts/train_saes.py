"""
CLI script for training Sparse Autoencoders on a SplitLlama model.

Usage:
    # Train SAEs on all residual stream hooks
    interp-train-saes \
        --model_path /path/to/split_llama_config_dir \
        --dataset_name togethercomputer/RedPajama-Data-1T-Sample \
        --hook_names layers_first.0.resid_post layers_first.1.resid_post \
        --expansion_factor 32 \
        --activation_fn topk \
        --k 32 \
        --total_tokens 100000000 \
        --batch_size 4096 \
        --checkpoint_dir ./checkpoints/saes

    # Train from cached activations
    interp-train-saes \
        --cache_dir ./act_cache \
        --hook_names layers_first.0.resid_post \
        --expansion_factor 32 \
        --checkpoint_dir ./checkpoints/saes
"""

from __future__ import annotations

import argparse
import os

import torch

from interp.models.split_llama import SplitLlama, SplitLlamaConfig
from interp.training.config import SAEConfig, TrainingConfig
from interp.training.sae import SAETrainer, SparseAutoencoder
from interp.wrapper.activation_store import ActivationStore
from interp.wrapper.hooked_model import HookedSplitLlama


def parse_args():
    p = argparse.ArgumentParser(description="Train SAEs on SplitLlama")

    # Model
    p.add_argument("--model_path", type=str, default="", help="Path to SplitLlama config dir")
    p.add_argument("--device", type=str, default="cuda")

    # Data source (either model+dataset or cached activations)
    p.add_argument("--dataset_name", type=str, default="")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--cache_dir", type=str, default="", help="Load from cached activations")
    p.add_argument("--context_len", type=int, default=1024)

    # Hook points
    p.add_argument("--hook_names", nargs="+", required=True)
    p.add_argument("--train_all_resid", action="store_true", help="Train on all residual stream hooks")

    # SAE config
    p.add_argument("--expansion_factor", type=int, default=32)
    p.add_argument("--activation_fn", type=str, default="topk", choices=["topk", "jumprelu", "relu"])
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--l1_coeff", type=float, default=1e-3)

    # Training
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--total_tokens", type=int, default=100_000_000)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints/saes")
    p.add_argument("--wandb_project", type=str, default="")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.cache_dir:
        # Train from cached activations
        _train_from_cache(args)
    elif args.model_path and args.dataset_name:
        # Train from model + dataset
        _train_from_model(args)
    else:
        raise ValueError(
            "Provide either --cache_dir (cached activations) or "
            "both --model_path and --dataset_name"
        )


def _train_from_cache(args):
    """Train SAEs from pre-cached activations."""
    for hook_name in args.hook_names:
        print(f"\n--- Training SAE for {hook_name} ---")

        # Determine d_in from cached metadata
        import json
        from pathlib import Path
        with open(Path(args.cache_dir) / "meta.json") as f:
            meta = json.load(f)

        d_in = meta["dims"][hook_name]

        sae_cfg = SAEConfig(
            d_in=d_in,
            expansion_factor=args.expansion_factor,
            activation_fn=args.activation_fn,
            k=args.k,
        )

        train_cfg = TrainingConfig(
            lr=args.lr,
            batch_size=args.batch_size,
            total_tokens=args.total_tokens,
            checkpoint_dir=os.path.join(args.checkpoint_dir, hook_name.replace(".", "_")),
            wandb_project=args.wandb_project,
            wandb_run_name=f"sae_{hook_name}",
            device=args.device,
            seed=args.seed,
        )

        sae = SparseAutoencoder(sae_cfg)
        trainer = SAETrainer(sae, train_cfg)

        # Create a multi-epoch loader
        def make_loader():
            tokens_yielded = 0
            while tokens_yielded < args.total_tokens:
                for batch in ActivationStore.get_cached_loader(
                    args.cache_dir,
                    hook_name,
                    batch_size=args.batch_size,
                    device=args.device,
                ):
                    yield batch
                    tokens_yielded += batch.shape[0]
                    if tokens_yielded >= args.total_tokens:
                        return

        trainer.train(make_loader())


def _train_from_model(args):
    """Train SAEs by extracting activations on-the-fly from the model."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Load model
    config = SplitLlamaConfig(model_path=args.model_path, context_len=args.context_len)
    model = SplitLlama(config)
    hooked = HookedSplitLlama(model, device=args.device)

    # Resolve hook names
    hook_names = args.hook_names
    if args.train_all_resid:
        hook_names = hooked.topology.all_resid_hooks

    # Load dataset
    dataset = load_dataset(args.dataset_name, split=args.dataset_split, streaming=True)

    # Detect tokenizer from model path
    import json
    with open(os.path.join(args.model_path, "config.json")) as f:
        model_cfg = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["path8b"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize dataset
    def tokenize_iter():
        for example in dataset:
            text = example[args.text_column]
            encoded = tokenizer(
                text,
                max_length=args.context_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            yield encoded["input_ids"].squeeze(0)

    # Train SAE for each hook
    for hook_name in hook_names:
        print(f"\n--- Training SAE for {hook_name} ---")

        d_in = hooked.get_d_model(hook_name)

        sae_cfg = SAEConfig(
            d_in=d_in,
            expansion_factor=args.expansion_factor,
            activation_fn=args.activation_fn,
            k=args.k,
        )

        train_cfg = TrainingConfig(
            lr=args.lr,
            batch_size=args.batch_size,
            total_tokens=args.total_tokens,
            checkpoint_dir=os.path.join(args.checkpoint_dir, hook_name.replace(".", "_")),
            wandb_project=args.wandb_project,
            wandb_run_name=f"sae_{hook_name}",
            device=args.device,
            seed=args.seed,
        )

        sae = SparseAutoencoder(sae_cfg)
        trainer = SAETrainer(sae, train_cfg)

        store = ActivationStore(hooked, hook_names=[hook_name], context_len=args.context_len)

        def make_activation_iter():
            for act_dict in store.stream(tokenize_iter(), batch_size=32):
                # act_dict[hook_name] is (B*T, d_in)
                acts = act_dict[hook_name]
                # Split into training-batch-sized chunks
                for i in range(0, acts.shape[0], args.batch_size):
                    yield acts[i : i + args.batch_size]

        trainer.train(make_activation_iter())


if __name__ == "__main__":
    main()
