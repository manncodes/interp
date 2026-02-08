"""Utilities for loading pre-tokenized binary data files."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Common filenames to search for in a tokenized directory
_BIN_FILENAMES = ["train.bin", "val.bin", "data.bin", "tokens.bin"]


def resolve_bin_path(tokenized_dir: str) -> Path:
    """Resolve a tokenized_dir to the actual .bin file path.

    Accepts either a direct file path or a directory containing a .bin file.
    """
    path = Path(tokenized_dir)
    if path.is_file():
        return path

    if path.is_dir():
        # Try common filenames
        for name in _BIN_FILENAMES:
            candidate = path / name
            if candidate.exists():
                return candidate
        # Try any .bin file
        bins = list(path.glob("*.bin"))
        if len(bins) == 1:
            return bins[0]
        elif len(bins) > 1:
            raise FileNotFoundError(
                f"Multiple .bin files in {path}: {[b.name for b in bins]}. "
                f"Pass the exact file path instead."
            )

    raise FileNotFoundError(
        f"No .bin file found at {path}. Expected a file or a directory "
        f"containing one of: {_BIN_FILENAMES}"
    )


def tokenized_bin_iter(
    tokenized_dir: str,
    context_len: int,
    token_dtype: str = "uint16",
    shuffle: bool = True,
    seed: int = 42,
):
    """Yield (context_len,) token ID tensors from a pre-tokenized .bin file.

    The .bin file is a flat array of token IDs stored as ``token_dtype``
    (typically uint16 for vocab < 65536, or int32/uint32 for larger vocabs).
    The file is memory-mapped so it doesn't need to fit in RAM.

    Args:
        tokenized_dir: Path to a .bin file or a directory containing one.
        context_len: Number of tokens per sequence.
        token_dtype: Numpy dtype of the stored token IDs.
        shuffle: Shuffle sequence order (not individual tokens).
        seed: Random seed for shuffling.

    Yields:
        torch.Tensor of shape (context_len,) with dtype torch.long.
    """
    bin_path = resolve_bin_path(tokenized_dir)
    data = np.memmap(str(bin_path), dtype=token_dtype, mode="r")
    n_tokens = len(data)
    n_sequences = n_tokens // context_len
    remainder = n_tokens % context_len

    file_size_mb = bin_path.stat().st_size / (1024 * 1024)

    logger.info("Tokenized binary data:")
    logger.info("  file:       %s (%.1f MB)", bin_path, file_size_mb)
    logger.info("  dtype:      %s", token_dtype)
    logger.info("  tokens:     %d", n_tokens)
    logger.info("  sequences:  %d (context_len=%d)", n_sequences, context_len)
    if remainder > 0:
        logger.info("  remainder:  %d tokens dropped", remainder)
    logger.info("  shuffle:    %s", shuffle)

    indices = np.arange(n_sequences)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

    for idx in indices:
        start = idx * context_len
        end = start + context_len
        chunk = np.array(data[start:end], dtype=np.int64)
        yield torch.from_numpy(chunk)
