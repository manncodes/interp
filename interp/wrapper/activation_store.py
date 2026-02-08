"""
Streaming activation extraction and disk-based caching.

Extracts activations from the model on a dataset, storing them as
memory-mapped numpy arrays for efficient SAE/transcoder training
without keeping the full model in memory during training.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from interp.wrapper.hooked_model import HookedSplitLlama

logger = logging.getLogger(__name__)


class ActivationStore:
    """
    Extracts and caches activations from a HookedSplitLlama model.

    Supports two modes:
    1. Streaming: yields activation batches on-the-fly (no disk usage)
    2. Cached: writes activations to disk as memory-mapped files,
       then loads them efficiently during training

    Usage (streaming):
        store = ActivationStore(hooked_model, hook_names=["layers_first.0.mlp_out"])
        for batch in store.stream(dataset, batch_size=32):
            # batch["layers_first.0.mlp_out"] is a tensor of shape (B*T, d_model)
            ...

    Usage (cached):
        store = ActivationStore(hooked_model, hook_names=["layers_first.0.mlp_out"])
        store.cache_to_disk(dataset, cache_dir="./act_cache", batch_size=32)
        loader = store.get_cached_loader(cache_dir="./act_cache", batch_size=4096)
        for batch in loader:
            ...
    """

    def __init__(
        self,
        hooked_model: HookedSplitLlama,
        hook_names: list[str],
        context_len: int = 1024,
    ):
        self.hooked_model = hooked_model
        self.hook_names = hook_names
        self.context_len = context_len
        self._dims = {
            name: hooked_model.get_d_model(name) for name in hook_names
        }

        logger.info(
            "ActivationStore initialized: context_len=%d, hooks=%d",
            self.context_len,
            len(self.hook_names),
        )
        for name in self.hook_names:
            logger.info("  hook: %-40s  dim=%d", name, self._dims[name])

    def _tokenize_batch(self, texts: list[str], tokenizer) -> torch.Tensor:
        """Tokenize and pad/truncate a batch of texts."""
        encoded = tokenizer(
            texts,
            max_length=self.context_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return encoded["input_ids"]

    def stream(
        self,
        token_ids_iter,
        batch_size: int = 32,
        flatten: bool = True,
    ):
        """
        Yield activation batches from an iterator of token ID tensors.

        Args:
            token_ids_iter: Iterator yielding token ID tensors of shape
                (seq_len,) or (batch, seq_len).
            batch_size: Number of sequences per batch.
            flatten: If True, reshape (B, T, D) -> (B*T, D) for SAE training.

        Yields:
            Dict mapping hook names to activation tensors.
        """
        logger.info(
            "stream() started: batch_size=%d, flatten=%s",
            batch_size,
            flatten,
        )
        t_start = time.time()
        sequences_processed = 0
        batches_yielded = 0
        buffer = []

        for token_ids in token_ids_iter:
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)
            buffer.append(token_ids)

            if len(buffer) >= batch_size:
                batch_ids = torch.cat(buffer[:batch_size], dim=0)
                buffer = buffer[batch_size:]

                acts = self.hooked_model.run_with_cache(
                    batch_ids, hook_names=self.hook_names
                )

                if flatten:
                    acts = {
                        k: v.reshape(-1, v.shape[-1]) for k, v in acts.items()
                    }

                sequences_processed += batch_ids.shape[0]
                batches_yielded += 1

                if batches_yielded % 50 == 0:
                    elapsed = time.time() - t_start
                    logger.info(
                        "stream() progress: %d sequences processed, "
                        "%d batches yielded (%.1f seq/s)",
                        sequences_processed,
                        batches_yielded,
                        sequences_processed / max(elapsed, 1e-6),
                    )

                yield acts

        # Flush remaining
        if buffer:
            batch_ids = torch.cat(buffer, dim=0)
            acts = self.hooked_model.run_with_cache(
                batch_ids, hook_names=self.hook_names
            )
            if flatten:
                acts = {
                    k: v.reshape(-1, v.shape[-1]) for k, v in acts.items()
                }
            sequences_processed += batch_ids.shape[0]
            batches_yielded += 1
            yield acts

        elapsed = time.time() - t_start
        logger.info(
            "stream() finished: %d sequences in %d batches, "
            "%.1f seconds (%.1f seq/s)",
            sequences_processed,
            batches_yielded,
            elapsed,
            sequences_processed / max(elapsed, 1e-6),
        )

    def cache_to_disk(
        self,
        token_ids_iter,
        cache_dir: str,
        batch_size: int = 32,
        max_tokens: int | None = None,
        dtype: str = "float32",
    ) -> dict[str, str]:
        """
        Extract activations and write to disk as memory-mapped numpy arrays.

        Args:
            token_ids_iter: Iterator yielding token ID tensors.
            cache_dir: Directory to write cached activations.
            batch_size: Sequences per forward pass.
            max_tokens: Stop after this many tokens (None = exhaust iterator).
            dtype: Numpy dtype for stored activations.

        Returns:
            Dict mapping hook names to file paths of cached .npy files.
        """
        logger.info(
            "cache_to_disk() started: cache_dir=%s, batch_size=%d, "
            "max_tokens=%s, dtype=%s",
            cache_dir,
            batch_size,
            max_tokens if max_tokens is not None else "unlimited",
            dtype,
        )

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # First pass: collect activations into lists, then write
        collected: dict[str, list[np.ndarray]] = {
            name: [] for name in self.hook_names
        }
        total_tokens = 0

        t_start = time.time()
        pbar = tqdm(
            desc="Caching activations",
            unit="tok",
            unit_scale=True,
            smoothing=0.1,
        )
        for acts in self.stream(token_ids_iter, batch_size=batch_size, flatten=True):
            for name in self.hook_names:
                arr = acts[name].cpu().float().numpy().astype(dtype)
                collected[name].append(arr)
                if name == self.hook_names[0]:
                    chunk_tokens = arr.shape[0]
                    total_tokens += chunk_tokens
                    pbar.update(chunk_tokens)

                    elapsed = time.time() - t_start
                    throughput = total_tokens / max(elapsed, 1e-6)
                    pbar.set_postfix(
                        tokens=total_tokens,
                        throughput=f"{throughput:.0f} tok/s",
                    )

            if max_tokens and total_tokens >= max_tokens:
                logger.info(
                    "Reached max_tokens limit (%d), stopping collection",
                    max_tokens,
                )
                break

        pbar.close()

        collection_time = time.time() - t_start
        logger.info(
            "Activation collection complete: %d tokens in %.1f seconds "
            "(%.0f tok/s)",
            total_tokens,
            collection_time,
            total_tokens / max(collection_time, 1e-6),
        )

        # Concatenate and write memory-mapped files
        t_write_start = time.time()
        file_paths = {}
        for name in self.hook_names:
            all_acts = np.concatenate(collected[name], axis=0)
            if max_tokens:
                all_acts = all_acts[:max_tokens]

            safe_name = name.replace(".", "_")
            fpath = cache_path / f"{safe_name}.npy"
            logger.info(
                "Writing %s: shape=%s, dtype=%s -> %s",
                name,
                all_acts.shape,
                all_acts.dtype,
                fpath,
            )
            np.save(str(fpath), all_acts)

            file_size_mb = os.path.getsize(str(fpath)) / (1024 * 1024)
            logger.info(
                "Wrote %s (%.2f MB)",
                fpath,
                file_size_mb,
            )
            file_paths[name] = str(fpath)

        write_time = time.time() - t_write_start

        # Write metadata
        meta = {
            "hook_names": self.hook_names,
            "dims": self._dims,
            "total_tokens": int(all_acts.shape[0]),
            "dtype": dtype,
            "files": file_paths,
        }
        meta_path = cache_path / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Wrote metadata to %s", meta_path)

        total_time = time.time() - t_start
        total_file_size_mb = sum(
            os.path.getsize(fp) / (1024 * 1024) for fp in file_paths.values()
        )
        logger.info(
            "cache_to_disk() summary: %d total tokens, %d hook(s), "
            "%.2f MB total on disk, collection=%.1fs, write=%.1fs, "
            "total=%.1fs",
            int(all_acts.shape[0]),
            len(self.hook_names),
            total_file_size_mb,
            collection_time,
            write_time,
            total_time,
        )

        return file_paths

    @staticmethod
    def get_cached_loader(
        cache_dir: str,
        hook_name: str,
        batch_size: int = 4096,
        shuffle: bool = True,
        device: str = "cpu",
    ):
        """
        Create a generator that yields batches from cached activations.

        Args:
            cache_dir: Directory containing cached .npy files.
            hook_name: Which hook's activations to load.
            batch_size: Number of activation vectors per batch.
            shuffle: Whether to shuffle the data each epoch.
            device: Device to move tensors to.

        Yields:
            Activation tensors of shape (batch_size, d_model).
        """
        cache_path = Path(cache_dir)
        with open(cache_path / "meta.json") as f:
            meta = json.load(f)

        fpath = meta["files"][hook_name]
        data = np.load(fpath, mmap_mode="r")
        n = data.shape[0]
        total_batches = (n + batch_size - 1) // batch_size

        logger.info(
            "get_cached_loader(): hook=%s, shape=%s, batch_size=%d, "
            "total_batches=%d, shuffle=%s, device=%s",
            hook_name,
            data.shape,
            batch_size,
            total_batches,
            shuffle,
            device,
        )

        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            batch = torch.tensor(
                np.array(data[idx]),  # copy from mmap
                dtype=torch.float32,
                device=device,
            )
            yield batch

    @staticmethod
    def get_cached_pair_loader(
        cache_dir: str,
        hook_name_in: str,
        hook_name_out: str,
        batch_size: int = 4096,
        shuffle: bool = True,
        device: str = "cpu",
    ):
        """
        Yield paired (input, output) activation batches for transcoder training.

        Args:
            cache_dir: Directory containing cached .npy files.
            hook_name_in: MLP input hook name.
            hook_name_out: MLP output hook name.
            batch_size: Batch size.
            shuffle: Shuffle data.
            device: Target device.

        Yields:
            Tuple of (input_acts, output_acts) tensors.
        """
        cache_path = Path(cache_dir)
        with open(cache_path / "meta.json") as f:
            meta = json.load(f)

        data_in = np.load(meta["files"][hook_name_in], mmap_mode="r")
        data_out = np.load(meta["files"][hook_name_out], mmap_mode="r")
        assert data_in.shape[0] == data_out.shape[0]

        n = data_in.shape[0]
        total_batches = (n + batch_size - 1) // batch_size

        logger.info(
            "get_cached_pair_loader(): hook_in=%s shape=%s, "
            "hook_out=%s shape=%s, batch_size=%d, total_batches=%d, "
            "shuffle=%s, device=%s",
            hook_name_in,
            data_in.shape,
            hook_name_out,
            data_out.shape,
            batch_size,
            total_batches,
            shuffle,
            device,
        )

        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            batch_in = torch.tensor(
                np.array(data_in[idx]),
                dtype=torch.float32,
                device=device,
            )
            batch_out = torch.tensor(
                np.array(data_out[idx]),
                dtype=torch.float32,
                device=device,
            )
            yield batch_in, batch_out
