"""
Sparse Autoencoder model and training loop.

Supports TopK, JumpReLU, and standard ReLU activation functions.
Includes dead feature resampling and decoder normalization.
Supports distributed training with PyTorch DDP via torchrun.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from interp.training.config import SAEConfig, TrainingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int, int]:
    """
    Initialize the PyTorch distributed process group for DDP training.

    Should be called at the start of each worker process launched by torchrun.
    Uses the NCCL backend for GPU communication and reads LOCAL_RANK from the
    environment to set the correct CUDA device.

    Returns:
        (rank, local_rank, world_size)
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), local_rank, dist.get_world_size()


def cleanup_distributed():
    """Destroy the distributed process group if one is active."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# SparseAutoencoder model (unchanged)
# ---------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder that decomposes activations into interpretable features.

    Architecture:
        encode: x -> activation_fn(W_enc @ (x - b_dec) + b_enc)
        decode: h -> W_dec @ h + b_dec
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        d_in = cfg.d_in
        d_sae = cfg.d_sae

        self.W_enc = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Track feature activation frequency for dead neuron detection
        self.register_buffer(
            "feature_act_count", torch.zeros(d_sae, dtype=torch.long)
        )
        self.register_buffer("total_batches", torch.tensor(0, dtype=torch.long))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc)
        if self.cfg.tied_init:
            self.W_dec.data = self.W_enc.data.T.clone()
        else:
            nn.init.kaiming_uniform_(self.W_dec)

        if self.cfg.normalize_decoder:
            self._normalize_decoder()

    def _normalize_decoder(self):
        with torch.no_grad():
            norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.W_dec.data /= norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        pre_acts = x_centered @ self.W_enc.T + self.b_enc
        return self._activate(pre_acts)

    def _activate(self, pre_acts: torch.Tensor) -> torch.Tensor:
        if self.cfg.activation_fn == "topk":
            return self._topk(pre_acts)
        elif self.cfg.activation_fn == "jumprelu":
            return self._jumprelu(pre_acts)
        elif self.cfg.activation_fn == "relu":
            return F.relu(pre_acts)
        else:
            raise ValueError(f"Unknown activation: {self.cfg.activation_fn}")

    def _topk(self, pre_acts: torch.Tensor) -> torch.Tensor:
        topk_vals, topk_idx = pre_acts.topk(self.cfg.k, dim=-1)
        topk_vals = F.relu(topk_vals)  # ensure non-negative
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_idx, topk_vals)
        return acts

    def _jumprelu(self, pre_acts: torch.Tensor) -> torch.Tensor:
        mask = (pre_acts > self.cfg.jumprelu_threshold).float()
        return pre_acts * mask

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.W_dec.T + self.b_dec

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input activations, shape (..., d_in)

        Returns:
            x_hat: Reconstructed activations
            h: Sparse feature activations
            loss: Combined reconstruction + sparsity loss
        """
        h = self.encode(x)
        x_hat = self.decode(h)

        # Reconstruction loss
        recon_loss = F.mse_loss(x_hat, x)

        # Sparsity loss (only needed for ReLU; TopK/JumpReLU have built-in sparsity)
        if self.cfg.activation_fn == "relu":
            sparsity_loss = h.abs().mean()
        else:
            sparsity_loss = torch.tensor(0.0, device=x.device)

        loss = recon_loss + self.cfg.l1_coeff * sparsity_loss if hasattr(self.cfg, 'l1_coeff') else recon_loss

        # Track feature activations
        if self.training:
            self.feature_act_count += (h > 0).sum(dim=0).long()
            self.total_batches += 1

        return x_hat, h, loss

    @property
    def dead_features(self) -> torch.Tensor:
        """Boolean mask of features that haven't activated recently."""
        if self.total_batches == 0:
            return torch.zeros(self.cfg.d_sae, dtype=torch.bool)
        avg_freq = self.feature_act_count.float() / (
            self.total_batches * self.cfg.k
            if self.cfg.activation_fn == "topk"
            else self.total_batches
        )
        return avg_freq < 1e-7

    def resample_dead_features(self, data_sample: torch.Tensor):
        """Reinitialize dead features using high-loss examples."""
        dead_mask = self.dead_features
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return 0

        with torch.no_grad():
            # Find high-reconstruction-error examples
            x_hat = self.decode(self.encode(data_sample))
            losses = (data_sample - x_hat).pow(2).sum(dim=-1)
            _, top_idx = losses.topk(min(n_dead, len(losses)))

            # Reinitialize dead encoder rows with high-loss examples
            replacement = data_sample[top_idx]
            replacement = replacement / replacement.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            dead_indices = dead_mask.nonzero(as_tuple=True)[0][:len(top_idx)]
            self.W_enc.data[dead_indices] = replacement[:len(dead_indices)] * 0.2
            self.W_dec.data[:, dead_indices] = (
                replacement[:len(dead_indices)].T * 0.2
            )
            self.b_enc.data[dead_indices] = 0.0
            self.feature_act_count[dead_indices] = 0

        return n_dead

    def save(self, path: str):
        """Save model weights and config."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        save_file(
            {k: v for k, v in self.state_dict().items() if not k.startswith("feature_act") and k != "total_batches"},
            str(save_dir / "sae_weights.safetensors"),
        )

        with open(save_dir / "cfg.json", "w") as f:
            json.dump(vars(self.cfg), f, indent=2)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> SparseAutoencoder:
        """Load a saved SAE."""
        save_dir = Path(path)
        with open(save_dir / "cfg.json") as f:
            cfg_dict = json.load(f)

        cfg = SAEConfig(**cfg_dict)
        sae = cls(cfg)

        weights = load_file(str(save_dir / "sae_weights.safetensors"), device=device)
        sae.load_state_dict(weights, strict=False)
        return sae


def _format_time(seconds: float) -> str:
    """Format seconds into a human-readable string (e.g. 1h 23m 45s)."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins:02d}m {secs:02d}s"


def _count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SAETrainer:
    """Training loop for sparse autoencoders.

    Supports single-GPU and multi-GPU (DDP) training. When ``distributed``
    is True (auto-detected from ``torch.distributed.is_initialized()`` or set
    explicitly via ``TrainingConfig.distributed``), the SAE is wrapped in
    ``DistributedDataParallel`` and logging / checkpointing / wandb are gated
    to rank 0 only.
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        train_cfg: TrainingConfig,
    ):
        self.sae = sae
        self.cfg = train_cfg

        # ---- Distributed detection ----
        self.distributed: bool = train_cfg.distributed or dist.is_initialized()
        if self.distributed and not dist.is_initialized():
            setup_distributed()

        self.rank: int = dist.get_rank() if self.distributed else 0
        self.local_rank: int = (
            int(os.environ.get("LOCAL_RANK", 0)) if self.distributed else 0
        )
        self.world_size: int = dist.get_world_size() if self.distributed else 1
        self.is_main: bool = self.rank == 0

        # ---- Device assignment ----
        if self.distributed:
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device(train_cfg.device)

        self.sae.to(self.device)

        # ---- Wrap model with DDP when distributed ----
        if self.distributed:
            self.ddp_sae = nn.parallel.DistributedDataParallel(
                self.sae,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )
        else:
            self.ddp_sae = None

        # The module used for the forward pass: DDP wrapper or raw model
        self._forward_model: nn.Module = self.ddp_sae if self.distributed else self.sae

        self.optimizer = torch.optim.Adam(
            self.sae.parameters(),
            lr=train_cfg.lr,
            betas=(train_cfg.beta1, train_cfg.beta2),
            weight_decay=train_cfg.weight_decay,
        )

        self._step = 0
        self._tokens_seen = 0
        self._wandb_run = None

    # ------------------------------------------------------------------
    # Distributed utilities
    # ------------------------------------------------------------------

    def _sync_loss(self, loss_value: torch.Tensor) -> torch.Tensor:
        """Average a scalar loss tensor across all ranks.

        Returns the averaged tensor. In non-distributed mode this is a no-op.
        """
        if not self.distributed:
            return loss_value
        reduced = loss_value.detach().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= self.world_size
        return reduced

    # ------------------------------------------------------------------
    # Logging helpers (gated behind rank 0 in distributed mode)
    # ------------------------------------------------------------------

    def _init_wandb(self):
        if not self.is_main:
            return
        if self.cfg.wandb_project:
            import wandb

            self._wandb_run = wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.wandb_run_name or None,
                config={
                    "sae": vars(self.sae.cfg),
                    "training": vars(self.cfg),
                },
            )

    def _lr_schedule(self) -> float:
        """Linear warmup then constant."""
        if self._tokens_seen < self.cfg.warmup_tokens:
            return self._tokens_seen / max(self.cfg.warmup_tokens, 1)
        return 1.0

    def _log_startup_info(self):
        """Log full configuration and model summary at training start."""
        if not self.is_main:
            return

        sae_cfg = self.sae.cfg
        train_cfg = self.cfg
        param_count = _count_parameters(self.sae)

        logger.info("=" * 70)
        logger.info("SAE TRAINING -- STARTUP")
        logger.info("=" * 70)
        logger.info("Model configuration:")
        logger.info("  d_in:              %d", sae_cfg.d_in)
        logger.info("  d_sae:             %d", sae_cfg.d_sae)
        logger.info("  expansion_factor:  %d", sae_cfg.expansion_factor)
        logger.info("  activation_fn:     %s", sae_cfg.activation_fn)
        if sae_cfg.activation_fn == "topk":
            logger.info("  k:                 %d", sae_cfg.k)
        elif sae_cfg.activation_fn == "jumprelu":
            logger.info("  jumprelu_thresh:   %.4f", sae_cfg.jumprelu_threshold)
        logger.info("  normalize_decoder: %s", sae_cfg.normalize_decoder)
        logger.info("  tied_init:         %s", sae_cfg.tied_init)
        logger.info("  trainable params:  %s", f"{param_count:,}")
        logger.info("Training configuration:")
        logger.info("  device:            %s", self.device)
        logger.info("  dtype:             %s", train_cfg.dtype)
        logger.info("  lr:                %.2e", train_cfg.lr)
        logger.info("  weight_decay:      %.2e", train_cfg.weight_decay)
        logger.info("  batch_size:        %d", train_cfg.batch_size)
        logger.info("  total_tokens:      %s", f"{train_cfg.total_tokens:,}")
        logger.info("  warmup_tokens:     %s", f"{train_cfg.warmup_tokens:,}")
        logger.info("  log_every:         %d steps", train_cfg.log_every)
        logger.info("  checkpoint_every:  %s tokens", f"{train_cfg.checkpoint_every:,}")
        logger.info("  checkpoint_dir:    %s", train_cfg.checkpoint_dir or "(none)")
        logger.info("  resample_dead:     %s (every %d steps)", train_cfg.resample_dead, train_cfg.resample_every)
        logger.info("  wandb_project:     %s", train_cfg.wandb_project or "(disabled)")
        if self.distributed:
            logger.info("  distributed:       True (world_size=%d)", self.world_size)
        else:
            logger.info("  distributed:       False")
        if torch.cuda.is_available():
            logger.info("  GPU:               %s", torch.cuda.get_device_name(self.local_rank))
            gpu_mem = torch.cuda.get_device_properties(self.local_rank).total_mem / (1024 ** 3)
            logger.info("  GPU memory:        %.1f GB", gpu_mem)
        logger.info("=" * 70)

    def _log_periodic_summary(
        self,
        loss: float,
        recon_loss: float,
        cosine_sim: float,
        variance_explained: float,
        l0: float,
        n_dead: int,
        lr_mult: float,
        elapsed: float,
    ):
        """Log a rich summary block every N steps."""
        if not self.is_main:
            return

        throughput = self._tokens_seen / max(elapsed, 1e-9)
        dead_pct = 100.0 * n_dead / self.sae.cfg.d_sae

        logger.info("-" * 60)
        logger.info(
            "Step %-8d | Tokens: %s / %s (%.1f%%)",
            self._step,
            f"{self._tokens_seen:,}",
            f"{self.cfg.total_tokens:,}",
            100.0 * self._tokens_seen / max(self.cfg.total_tokens, 1),
        )
        logger.info(
            "  loss:          %.6f  |  recon_loss:     %.6f",
            loss,
            recon_loss,
        )
        logger.info(
            "  cosine_sim:    %.6f  |  var_explained:  %.6f",
            cosine_sim,
            variance_explained,
        )
        logger.info(
            "  L0 sparsity:   %.2f   |  dead features:  %d / %d (%.2f%%)",
            l0,
            n_dead,
            self.sae.cfg.d_sae,
            dead_pct,
        )
        logger.info(
            "  lr:            %.2e  |  throughput:     %.0f tok/s",
            self.cfg.lr * lr_mult,
            throughput,
        )
        logger.info(
            "  elapsed:       %s",
            _format_time(elapsed),
        )
        logger.info("-" * 60)

    def _log_final_summary(self, elapsed: float):
        """Log summary at the end of training."""
        if not self.is_main:
            return

        throughput = self._tokens_seen / max(elapsed, 1e-9)
        n_dead = self.sae.dead_features.sum().item()
        dead_pct = 100.0 * n_dead / self.sae.cfg.d_sae

        logger.info("=" * 70)
        logger.info("SAE TRAINING -- COMPLETE")
        logger.info("=" * 70)
        logger.info("  total steps:       %d", self._step)
        logger.info("  total tokens:      %s", f"{self._tokens_seen:,}")
        logger.info("  wall time:         %s", _format_time(elapsed))
        logger.info("  avg throughput:    %.0f tok/s", throughput)
        logger.info("  final dead feats:  %d / %d (%.2f%%)", n_dead, self.sae.cfg.d_sae, dead_pct)
        if self.cfg.checkpoint_dir:
            logger.info("  final checkpoint:  %s", os.path.join(self.cfg.checkpoint_dir, "sae_final"))
        logger.info("=" * 70)

    def train(self, activation_iter):
        """
        Train the SAE on an iterator of activation batches.

        Args:
            activation_iter: Iterator yielding tensors of shape (batch, d_in).
                Can be from ActivationStore.stream() or
                ActivationStore.get_cached_loader().
        """
        self._init_wandb()
        self._log_startup_info()
        self.sae.train()

        train_start = time.time()
        last_log_time = train_start

        # Only show tqdm progress bar on rank 0
        if self.is_main:
            pbar = tqdm(
                total=self.cfg.total_tokens,
                desc="Training SAE",
                unit="tok",
                unit_scale=True,
                smoothing=0.05,
            )
        else:
            pbar = None

        for batch in activation_iter:
            if self._tokens_seen >= self.cfg.total_tokens:
                break

            batch = batch.to(self.device)
            batch_tokens = batch.shape[0]

            # LR warmup
            lr_mult = self._lr_schedule()
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.cfg.lr * lr_mult

            # Forward -- use DDP-wrapped model when distributed
            x_hat, h, loss = self._forward_model(batch)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Normalize decoder columns (always on the underlying module)
            if self.sae.cfg.normalize_decoder:
                self.sae._normalize_decoder()

            self._step += 1
            self._tokens_seen += batch_tokens
            if pbar is not None:
                pbar.update(batch_tokens)

            # Logging
            if self._step % self.cfg.log_every == 0:
                now = time.time()
                elapsed = now - train_start

                with torch.no_grad():
                    l0 = (h > 0).float().sum(dim=-1).mean().item()
                    recon_loss = F.mse_loss(x_hat, batch).item()
                    cosine_sim = F.cosine_similarity(x_hat, batch, dim=-1).mean().item()
                    n_dead = self.sae.dead_features.sum().item()

                    # Variance explained: 1 - Var(x - x_hat) / Var(x)
                    residual_var = (batch - x_hat).var().item()
                    input_var = batch.var().item()
                    variance_explained = 1.0 - residual_var / max(input_var, 1e-12)

                # Sync loss across ranks so the logged value is the global mean
                if self.distributed:
                    loss_synced = self._sync_loss(loss)
                    loss_val = loss_synced.item()
                else:
                    loss_val = loss.item()

                throughput = self._tokens_seen / max(elapsed, 1e-9)

                if self.is_main:
                    metrics = {
                        "loss": loss_val,
                        "recon_loss": recon_loss,
                        "l0": l0,
                        "cosine_sim": cosine_sim,
                        "variance_explained": variance_explained,
                        "dead_features": n_dead,
                        "lr": self.cfg.lr * lr_mult,
                        "tokens_seen": self._tokens_seen,
                        "throughput_tok_s": throughput,
                    }

                    if self._wandb_run:
                        import wandb
                        wandb.log(metrics, step=self._step)

                    pbar.set_postfix(
                        loss=f"{loss_val:.4f}",
                        l0=f"{l0:.1f}",
                        cos=f"{cosine_sim:.4f}",
                        dead=n_dead,
                        tok_s=f"{throughput:.0f}",
                    )

                # Rich periodic summary (logger-based, gated behind is_main)
                self._log_periodic_summary(
                    loss=loss_val if self.is_main else loss.item(),
                    recon_loss=recon_loss,
                    cosine_sim=cosine_sim,
                    variance_explained=variance_explained,
                    l0=l0,
                    n_dead=n_dead,
                    lr_mult=lr_mult,
                    elapsed=elapsed,
                )
                last_log_time = now

            # Resample dead features
            if (
                self.cfg.resample_dead
                and self._step % self.cfg.resample_every == 0
                and self._step > 0
            ):
                n_resampled = self.sae.resample_dead_features(batch)
                if self.is_main:
                    if n_resampled > 0:
                        logger.info(
                            "[step %d] Resampled %d dead features (%.2f%% of d_sae=%d)",
                            self._step,
                            n_resampled,
                            100.0 * n_resampled / self.sae.cfg.d_sae,
                            self.sae.cfg.d_sae,
                        )
                    else:
                        logger.info(
                            "[step %d] Dead feature resampling check: 0 dead features found, nothing to resample",
                            self._step,
                        )
                    self.sae.feature_act_count.zero_()
                    self.sae.total_batches.zero_()

            # Checkpoint
            if (
                self.cfg.checkpoint_dir
                and self._tokens_seen % self.cfg.checkpoint_every < batch_tokens
            ):
                if self.is_main:
                    ckpt_path = os.path.join(
                        self.cfg.checkpoint_dir,
                        f"sae_step{self._step}",
                    )
                    self.sae.save(ckpt_path)
                    logger.info(
                        "[step %d] Checkpoint saved to: %s",
                        self._step,
                        ckpt_path,
                    )
                # All ranks wait until the checkpoint write is complete
                if self.distributed:
                    dist.barrier()

        if pbar is not None:
            pbar.close()

        total_elapsed = time.time() - train_start

        # Final save
        if self.cfg.checkpoint_dir:
            if self.is_main:
                final_path = os.path.join(self.cfg.checkpoint_dir, "sae_final")
                self.sae.save(final_path)
                logger.info("Final model saved to: %s", final_path)
            if self.distributed:
                dist.barrier()

        self._log_final_summary(total_elapsed)

        if self._wandb_run:
            import wandb
            wandb.finish()
