"""
CLI script for training transcoders on a SplitLlama model.

Transcoders map MLP inputs to MLP outputs through sparse features,
enabling weights-based circuit analysis.

Usage:
    # Train transcoders for all MLP layers
    interp-train-transcoders \
        --model_path /path/to/split_llama_config_dir \
        --dataset_name togethercomputer/RedPajama-Data-1T-Sample \
        --expansion_factor 32 \
        --activation_fn topk \
        --k 64 \
        --total_tokens 200000000 \
        --checkpoint_dir ./checkpoints/transcoders

    # Train from cached activations (must have cached both mlp_in and mlp_out)
    interp-train-transcoders \
        --cache_dir ./act_cache \
        --layers layers_first.0 layers_first.1 \
        --checkpoint_dir ./checkpoints/transcoders

    # Train adapter transcoder (bridges 8B -> 70B dimensions)
    interp-train-transcoders \
        --model_path /path/to/config_dir \
        --dataset_name dataset \
        --train_adapter \
        --checkpoint_dir ./checkpoints/transcoders
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from interp.models.split_llama import SplitLlama, SplitLlamaConfig
from interp.training.config import TrainingConfig, TranscoderConfig
from interp.training.transcoder import Transcoder, TranscoderTrainer
from interp.wrapper.activation_store import ActivationStore
from interp.wrapper.hooked_model import HookedSplitLlama


def parse_args():
    p = argparse.ArgumentParser(description="Train transcoders on SplitLlama")

    # Model
    p.add_argument("--model_path", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")

    # Data
    p.add_argument("--dataset_name", type=str, default="")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--cache_dir", type=str, default="")
    p.add_argument("--context_len", type=int, default=1024)

    # Layer selection
    p.add_argument("--layers", nargs="+", default=[], help="Layer keys (e.g., layers_first.0)")
    p.add_argument("--train_all", action="store_true", help="Train for all MLP layers")
    p.add_argument("--train_adapter", action="store_true", help="Train adapter transcoder")

    # Transcoder config
    p.add_argument("--expansion_factor", type=int, default=32)
    p.add_argument("--activation_fn", type=str, default="topk")
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--has_skip", action="store_true", default=True)
    p.add_argument("--no_skip", action="store_true", help="Disable skip connection")

    # Training
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--total_tokens", type=int, default=200_000_000)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints/transcoders")
    p.add_argument("--wandb_project", type=str, default="")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    has_skip = not args.no_skip

    if args.cache_dir:
        _train_from_cache(args, has_skip)
    elif args.model_path and args.dataset_name:
        _train_from_model(args, has_skip)
    else:
        raise ValueError(
            "Provide either --cache_dir or both --model_path and --dataset_name"
        )


def _get_mlp_pairs_for_layers(
    hooked: HookedSplitLlama, layers: list[str]
) -> list[tuple[str, str, str]]:
    """Return (layer_key, mlp_in_hook, mlp_out_hook) for specified layers."""
    all_pairs = hooked.get_mlp_pairs()
    result = []
    for mlp_in, mlp_out in all_pairs:
        layer_key = mlp_in.rsplit(".mlp_in", 1)[0]
        if not layers or layer_key in layers:
            result.append((layer_key, mlp_in, mlp_out))
    return result


def _train_from_cache(args, has_skip: bool):
    from pathlib import Path

    with open(Path(args.cache_dir) / "meta.json") as f:
        meta = json.load(f)

    # Find paired hooks in cache
    hook_names = meta["hook_names"]
    mlp_in_hooks = [h for h in hook_names if h.endswith(".mlp_in")]

    for mlp_in_hook in mlp_in_hooks:
        mlp_out_hook = mlp_in_hook.replace(".mlp_in", ".mlp_out")
        layer_key = mlp_in_hook.rsplit(".mlp_in", 1)[0]

        if args.layers and layer_key not in args.layers:
            continue

        if mlp_out_hook not in hook_names:
            print(f"Skipping {layer_key}: mlp_out not cached")
            continue

        d_in = meta["dims"][mlp_in_hook]
        d_out = meta["dims"][mlp_out_hook]

        print(f"\n--- Training transcoder for {layer_key} (d_in={d_in}, d_out={d_out}) ---")

        tc_cfg = TranscoderConfig(
            d_in=d_in,
            d_out=d_out,
            expansion_factor=args.expansion_factor,
            activation_fn=args.activation_fn,
            k=args.k,
            has_skip=has_skip,
        )

        train_cfg = TrainingConfig(
            lr=args.lr,
            batch_size=args.batch_size,
            total_tokens=args.total_tokens,
            checkpoint_dir=os.path.join(args.checkpoint_dir, layer_key.replace(".", "_")),
            wandb_project=args.wandb_project,
            wandb_run_name=f"tc_{layer_key}",
            device=args.device,
            seed=args.seed,
        )

        tc = Transcoder(tc_cfg)
        trainer = TranscoderTrainer(tc, train_cfg)

        def make_loader():
            tokens_yielded = 0
            while tokens_yielded < args.total_tokens:
                for batch_in, batch_out in ActivationStore.get_cached_pair_loader(
                    args.cache_dir,
                    mlp_in_hook,
                    mlp_out_hook,
                    batch_size=args.batch_size,
                    device=args.device,
                ):
                    yield batch_in, batch_out
                    tokens_yielded += batch_in.shape[0]
                    if tokens_yielded >= args.total_tokens:
                        return

        trainer.train(make_loader())


def _train_from_model(args, has_skip: bool):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    config = SplitLlamaConfig(model_path=args.model_path, context_len=args.context_len)
    model = SplitLlama(config)
    hooked = HookedSplitLlama(model, device=args.device)

    # Get MLP pairs to train
    if args.train_all:
        pairs = _get_mlp_pairs_for_layers(hooked, [])
    elif args.layers:
        pairs = _get_mlp_pairs_for_layers(hooked, args.layers)
    else:
        pairs = _get_mlp_pairs_for_layers(hooked, [])

    # Load dataset
    dataset = load_dataset(args.dataset_name, split=args.dataset_split, streaming=True)

    with open(os.path.join(args.model_path, "config.json")) as f:
        model_cfg = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["path8b"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    for layer_key, mlp_in_hook, mlp_out_hook in pairs:
        print(f"\n--- Training transcoder for {layer_key} ---")

        d_in = hooked.get_d_model(mlp_in_hook)
        d_out = hooked.get_d_model(mlp_out_hook)

        tc_cfg = TranscoderConfig(
            d_in=d_in,
            d_out=d_out,
            expansion_factor=args.expansion_factor,
            activation_fn=args.activation_fn,
            k=args.k,
            has_skip=has_skip,
        )

        train_cfg = TrainingConfig(
            lr=args.lr,
            batch_size=args.batch_size,
            total_tokens=args.total_tokens,
            checkpoint_dir=os.path.join(args.checkpoint_dir, layer_key.replace(".", "_")),
            wandb_project=args.wandb_project,
            wandb_run_name=f"tc_{layer_key}",
            device=args.device,
            seed=args.seed,
        )

        tc = Transcoder(tc_cfg)
        trainer = TranscoderTrainer(tc, train_cfg)

        store = ActivationStore(
            hooked,
            hook_names=[mlp_in_hook, mlp_out_hook],
            context_len=args.context_len,
        )

        def make_paired_iter():
            for act_dict in store.stream(tokenize_iter(), batch_size=32):
                acts_in = act_dict[mlp_in_hook]
                acts_out = act_dict[mlp_out_hook]
                for i in range(0, acts_in.shape[0], args.batch_size):
                    yield (
                        acts_in[i : i + args.batch_size],
                        acts_out[i : i + args.batch_size],
                    )

        trainer.train(make_paired_iter())


if __name__ == "__main__":
    main()
