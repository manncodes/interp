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

    # Multi-GPU distributed training
    torchrun --nproc_per_node=8 -m interp.scripts.train_saes \
        --model_path /path/to/config \
        --dataset_name <dataset> \
        --hook_names layers_first.0.resid_post \
        --checkpoint_dir ./checkpoints/saes
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time

import torch

from interp.models.split_llama import SplitLlama, SplitLlamaConfig
from interp.training.config import SAEConfig, TrainingConfig
from interp.training.sae import (
    SAETrainer,
    SparseAutoencoder,
    cleanup_distributed,
    setup_distributed,
)
from interp.wrapper.activation_store import ActivationStore
from interp.wrapper.hooked_model import HookedSplitLlama

logger = logging.getLogger("interp.scripts.train_saes")


def _log_environment(distributed: bool, rank: int, world_size: int):
    """Log system and environment info at startup."""
    logger.info("=" * 72)
    logger.info("ENVIRONMENT")
    logger.info("=" * 72)
    logger.info("  Python:        %s", sys.version.split()[0])
    logger.info("  Platform:      %s", platform.platform())
    logger.info("  PyTorch:       %s", torch.__version__)
    logger.info("  CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("  CUDA version:  %s", torch.version.cuda)
        logger.info("  GPU count:     %d", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                "  GPU %d:         %s (%.1f GB)",
                i, props.name, props.total_mem / (1024 ** 3),
            )
    if distributed:
        logger.info("  Distributed:   True (rank %d/%d)", rank, world_size)
    else:
        logger.info("  Distributed:   False")
    logger.info("=" * 72)


def parse_args():
    p = argparse.ArgumentParser(description="Train SAEs on SplitLlama")

    # Model
    p.add_argument("--model_path", type=str, default="", help="Path to SplitLlama config dir")
    p.add_argument("--device", type=str, default="cuda")

    # Data source (pick one: model+dataset, model+tokenized_dir, or cache_dir)
    p.add_argument("--dataset_name", type=str, default="", help="HuggingFace dataset name or local path")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--tokenized_dir", type=str, default="", help="Path to dir with train.bin (pre-tokenized)")
    p.add_argument("--token_dtype", type=str, default="uint16", help="Dtype of token IDs in .bin file (uint16, int32)")
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
    p.add_argument("--log_every", type=int, default=100, help="Log metrics every N steps")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints/saes")
    p.add_argument("--wandb_project", type=str, default="")
    p.add_argument("--seed", type=int, default=42)

    # Distributed
    p.add_argument("--distributed", action="store_true", help="Enable distributed training (auto-detected from LOCAL_RANK env var)")

    return p.parse_args()


def main():
    args = parse_args()

    # Auto-detect distributed mode from environment (set by torchrun)
    distributed = args.distributed or ("LOCAL_RANK" in os.environ)
    rank = 0
    local_rank = 0
    world_size = 1

    if distributed:
        rank, local_rank, world_size = setup_distributed()
        args.device = f"cuda:{local_rank}"

    # Configure logging: INFO on rank 0, WARNING on other ranks
    log_level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        force=True,
    )

    torch.manual_seed(args.seed)

    if rank == 0:
        _log_environment(distributed, rank, world_size)
        logger.info("CLI args: %s", vars(args))

    try:
        if args.cache_dir:
            _train_from_cache(args)
        elif args.model_path and (args.dataset_name or args.tokenized_dir):
            _train_from_model(args)
        else:
            raise ValueError(
                "Provide either --cache_dir, or --model_path with "
                "--dataset_name or --tokenized_dir"
            )
    finally:
        if distributed:
            cleanup_distributed()


def _train_from_cache(args):
    """Train SAEs from pre-cached activations."""
    from pathlib import Path

    logger.info("Loading cached activations from: %s", args.cache_dir)
    with open(Path(args.cache_dir) / "meta.json") as f:
        meta = json.load(f)

    logger.info("Cache metadata:")
    logger.info("  hooks: %s", meta["hook_names"])
    logger.info("  dims:  %s", meta["dims"])
    logger.info("  total_tokens: %s", meta.get("total_tokens", "unknown"))

    for hook_name in args.hook_names:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Training SAE for hook: %s", hook_name)
        logger.info("=" * 60)

        d_in = meta["dims"][hook_name]
        logger.info("  d_in = %d", d_in)

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
            log_every=args.log_every,
            checkpoint_dir=os.path.join(args.checkpoint_dir, hook_name.replace(".", "_")),
            wandb_project=args.wandb_project,
            wandb_run_name=f"sae_{hook_name}",
            device=args.device,
            seed=args.seed,
        )

        sae = SparseAutoencoder(sae_cfg)
        trainer = SAETrainer(sae, train_cfg)

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


def _build_token_iter(args):
    """Build a token ID iterator from either --tokenized_dir or --dataset_name."""
    if args.tokenized_dir:
        from interp.training.data import tokenized_bin_iter

        logger.info("Using pre-tokenized data from: %s", args.tokenized_dir)
        return lambda: tokenized_bin_iter(
            args.tokenized_dir,
            context_len=args.context_len,
            token_dtype=args.token_dtype,
            seed=args.seed,
        )

    from datasets import load_dataset
    from transformers import AutoTokenizer

    logger.info("Loading dataset: %s (split=%s, streaming=True)", args.dataset_name, args.dataset_split)
    t0 = time.time()
    dataset = load_dataset(args.dataset_name, split=args.dataset_split, streaming=True)
    logger.info("Dataset loaded in %.1fs", time.time() - t0)

    logger.info("Loading tokenizer from model config...")
    with open(os.path.join(args.model_path, "config.json")) as f:
        model_cfg = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["path8b"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded: %s (vocab_size=%d)", type(tokenizer).__name__, tokenizer.vocab_size)

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

    return tokenize_iter


def _train_from_model(args):
    """Train SAEs by extracting activations on-the-fly from the model."""
    # Load model
    logger.info("Loading model from: %s", args.model_path)
    t0 = time.time()
    config = SplitLlamaConfig(model_path=args.model_path, context_len=args.context_len)
    model = SplitLlama(config)
    hooked = HookedSplitLlama(model, device=args.device)
    logger.info("Model loaded in %.1fs", time.time() - t0)
    logger.info("  device: %s", args.device)
    logger.info("  context_len: %d", args.context_len)
    logger.info("  d_model_first: %d", hooked.topology.d_model_first)
    logger.info("  d_model_last: %d", hooked.topology.d_model_last)
    logger.info("  total resid hooks: %d", len(hooked.topology.all_resid_hooks))

    # Resolve hook names
    hook_names = args.hook_names
    if args.train_all_resid:
        hook_names = hooked.topology.all_resid_hooks
        logger.info("--train_all_resid: resolved to %d hooks", len(hook_names))
    logger.info("Hooks to train: %s", hook_names)

    # Build token iterator (from bin file or HF dataset)
    make_token_iter = _build_token_iter(args)

    # Train SAE for each hook
    for i, hook_name in enumerate(hook_names):
        logger.info("")
        logger.info("=" * 60)
        logger.info("Training SAE %d/%d: %s", i + 1, len(hook_names), hook_name)
        logger.info("=" * 60)

        d_in = hooked.get_d_model(hook_name)
        logger.info("  d_in = %d", d_in)
        logger.info("  d_sae = %d (expansion_factor=%d)", d_in * args.expansion_factor, args.expansion_factor)
        logger.info("  activation_fn = %s (k=%d)", args.activation_fn, args.k)

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
            log_every=args.log_every,
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
            for act_dict in store.stream(make_token_iter(), batch_size=32):
                acts = act_dict[hook_name]
                for i in range(0, acts.shape[0], args.batch_size):
                    yield acts[i : i + args.batch_size]

        trainer.train(make_activation_iter())

    logger.info("All %d SAE(s) trained.", len(hook_names))


if __name__ == "__main__":
    main()
