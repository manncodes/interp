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

    # Multi-GPU distributed training
    torchrun --nproc_per_node=8 -m interp.scripts.train_transcoders \
        --model_path /path/to/config \
        --dataset_name <dataset> \
        --layers layers_first.0 \
        --checkpoint_dir ./checkpoints/transcoders
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
from interp.training.config import TrainingConfig, TranscoderConfig
from interp.training.sae import cleanup_distributed, setup_distributed
from interp.training.transcoder import Transcoder, TranscoderTrainer
from interp.wrapper.activation_store import ActivationStore
from interp.wrapper.hooked_model import HookedSplitLlama

logger = logging.getLogger("interp.scripts.train_transcoders")


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
    p = argparse.ArgumentParser(description="Train transcoders on SplitLlama")

    # Model
    p.add_argument("--model_path", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")

    # Data source (pick one: model+dataset, model+tokenized_dir, or cache_dir)
    p.add_argument("--dataset_name", type=str, default="", help="HuggingFace dataset name or local path")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--tokenized_dir", type=str, default="", help="Path to dir with train.bin (pre-tokenized)")
    p.add_argument("--token_dtype", type=str, default="uint16", help="Dtype of token IDs in .bin file (uint16, int32)")
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
    p.add_argument("--log_every", type=int, default=100, help="Log metrics every N steps")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints/transcoders")
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

    has_skip = not args.no_skip

    if rank == 0:
        _log_environment(distributed, rank, world_size)
        logger.info("CLI args: %s", vars(args))

    try:
        if args.cache_dir:
            _train_from_cache(args, has_skip)
        elif args.model_path and (args.dataset_name or args.tokenized_dir):
            _train_from_model(args, has_skip)
        else:
            raise ValueError(
                "Provide either --cache_dir, or --model_path with "
                "--dataset_name or --tokenized_dir"
            )
    finally:
        if distributed:
            cleanup_distributed()


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

    logger.info("Loading cached activations from: %s", args.cache_dir)
    with open(Path(args.cache_dir) / "meta.json") as f:
        meta = json.load(f)

    # Find paired hooks in cache
    hook_names = meta["hook_names"]
    mlp_in_hooks = [h for h in hook_names if h.endswith(".mlp_in")]

    logger.info("Cache metadata:")
    logger.info("  hooks: %s", hook_names)
    logger.info("  dims:  %s", meta["dims"])
    logger.info("  MLP input hooks found: %d", len(mlp_in_hooks))

    trained_count = 0
    for mlp_in_hook in mlp_in_hooks:
        mlp_out_hook = mlp_in_hook.replace(".mlp_in", ".mlp_out")
        layer_key = mlp_in_hook.rsplit(".mlp_in", 1)[0]

        if args.layers and layer_key not in args.layers:
            logger.info("Skipping %s (not in --layers filter)", layer_key)
            continue

        if mlp_out_hook not in hook_names:
            logger.warning("Skipping %s: mlp_out not cached", layer_key)
            continue

        d_in = meta["dims"][mlp_in_hook]
        d_out = meta["dims"][mlp_out_hook]

        trained_count += 1
        logger.info("")
        logger.info("=" * 60)
        logger.info("Training transcoder for: %s", layer_key)
        logger.info("=" * 60)
        logger.info("  d_in = %d, d_out = %d", d_in, d_out)
        logger.info("  d_hidden = %d (expansion_factor=%d)", d_in * args.expansion_factor, args.expansion_factor)
        logger.info("  activation_fn = %s (k=%d)", args.activation_fn, args.k)
        logger.info("  has_skip = %s", has_skip)

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
            log_every=args.log_every,
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

    logger.info("All %d transcoder(s) trained.", trained_count)


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


def _train_from_model(args, has_skip: bool):
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

    # Get MLP pairs to train
    if args.train_all:
        pairs = _get_mlp_pairs_for_layers(hooked, [])
        logger.info("--train_all: training all %d MLP layers", len(pairs))
    elif args.layers:
        pairs = _get_mlp_pairs_for_layers(hooked, args.layers)
        logger.info("Training %d selected MLP layers: %s", len(pairs), args.layers)
    else:
        pairs = _get_mlp_pairs_for_layers(hooked, [])
        logger.info("No layer filter: training all %d MLP layers", len(pairs))

    for layer_key, mlp_in, mlp_out in pairs:
        logger.info("  %s: %s -> %s", layer_key, mlp_in, mlp_out)

    # Build token iterator (from bin file or HF dataset)
    make_token_iter = _build_token_iter(args)

    for i, (layer_key, mlp_in_hook, mlp_out_hook) in enumerate(pairs):
        d_in = hooked.get_d_model(mlp_in_hook)
        d_out = hooked.get_d_model(mlp_out_hook)

        logger.info("")
        logger.info("=" * 60)
        logger.info("Training transcoder %d/%d: %s", i + 1, len(pairs), layer_key)
        logger.info("=" * 60)
        logger.info("  mlp_in hook:  %s (d=%d)", mlp_in_hook, d_in)
        logger.info("  mlp_out hook: %s (d=%d)", mlp_out_hook, d_out)
        logger.info("  d_hidden = %d (expansion_factor=%d)", d_in * args.expansion_factor, args.expansion_factor)
        logger.info("  activation_fn = %s (k=%d)", args.activation_fn, args.k)
        logger.info("  has_skip = %s", has_skip)

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
            log_every=args.log_every,
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
            for act_dict in store.stream(make_token_iter(), batch_size=32):
                acts_in = act_dict[mlp_in_hook]
                acts_out = act_dict[mlp_out_hook]
                for i in range(0, acts_in.shape[0], args.batch_size):
                    yield (
                        acts_in[i : i + args.batch_size],
                        acts_out[i : i + args.batch_size],
                    )

        trainer.train(make_paired_iter())

    logger.info("All %d transcoder(s) trained.", len(pairs))


if __name__ == "__main__":
    main()
