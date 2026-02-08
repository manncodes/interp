"""
Transcoder model and training loop.

A transcoder maps MLP inputs to MLP outputs through a sparse
intermediate representation, enabling weights-based circuit analysis.

Supports:
- Per-layer transcoders (PLT): d_in == d_out (same hidden dim)
- Adapter transcoders: d_in != d_out (bridges 8B -> 70B dimensions)
- Skip transcoders: adds an affine skip connection for better fidelity
- Distributed training via PyTorch DDP (torchrun)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from interp.training.config import TranscoderConfig, TrainingConfig
from interp.training.sae import setup_distributed, cleanup_distributed

logger = logging.getLogger(__name__)


class Transcoder(nn.Module):
    """
    Transcoder that approximates an MLP layer with a sparse decomposition.

    Standard mode:
        encode: mlp_input -> activation_fn(W_enc @ mlp_input + b_enc)
        decode: h -> W_dec @ h + b_dec

    Skip mode (has_skip=True):
        output = W_dec @ h + b_dec + W_skip @ mlp_input + b_skip
    """

    def __init__(self, cfg: TranscoderConfig):
        super().__init__()
        self.cfg = cfg
        d_in = cfg.d_in
        d_out = cfg.d_out
        d_hidden = cfg.d_hidden

        self.W_enc = nn.Parameter(torch.empty(d_hidden, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.empty(d_out, d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_out))

        if cfg.has_skip:
            self.W_skip = nn.Parameter(torch.empty(d_out, d_in))
            self.b_skip = nn.Parameter(torch.zeros(d_out))

        self.register_buffer(
            "feature_act_count", torch.zeros(d_hidden, dtype=torch.long)
        )
        self.register_buffer("total_batches", torch.tensor(0, dtype=torch.long))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        if self.cfg.has_skip:
            # Initialize skip connection close to identity (or zero for dim mismatch)
            if self.cfg.d_in == self.cfg.d_out:
                nn.init.eye_(self.W_skip)
            else:
                nn.init.kaiming_uniform_(self.W_skip)
            nn.init.zeros_(self.b_skip)

    def encode(self, mlp_input: torch.Tensor) -> torch.Tensor:
        pre_acts = mlp_input @ self.W_enc.T + self.b_enc
        return self._activate(pre_acts)

    def _activate(self, pre_acts: torch.Tensor) -> torch.Tensor:
        if self.cfg.activation_fn == "topk":
            topk_vals, topk_idx = pre_acts.topk(self.cfg.k, dim=-1)
            topk_vals = F.relu(topk_vals)
            acts = torch.zeros_like(pre_acts)
            acts.scatter_(-1, topk_idx, topk_vals)
            return acts
        elif self.cfg.activation_fn == "jumprelu":
            mask = (pre_acts > self.cfg.jumprelu_threshold).float()
            return pre_acts * mask
        elif self.cfg.activation_fn == "relu":
            return F.relu(pre_acts)
        else:
            raise ValueError(f"Unknown activation: {self.cfg.activation_fn}")

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.W_dec.T + self.b_dec

    def forward(
        self, mlp_input: torch.Tensor, mlp_output_target: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mlp_input: MLP layer input activations, shape (..., d_in)
            mlp_output_target: True MLP output for computing loss, shape (..., d_out)

        Returns:
            mlp_output_hat: Predicted MLP output
            h: Sparse feature activations
            loss: Reconstruction loss
        """
        h = self.encode(mlp_input)
        mlp_output_hat = self.decode(h)

        if self.cfg.has_skip:
            mlp_output_hat = mlp_output_hat + mlp_input @ self.W_skip.T + self.b_skip

        # Loss
        if mlp_output_target is not None:
            loss = F.mse_loss(mlp_output_hat, mlp_output_target)
        else:
            loss = torch.tensor(0.0, device=mlp_input.device)

        if self.training:
            self.feature_act_count += (h > 0).sum(dim=0).long()
            self.total_batches += 1

        return mlp_output_hat, h, loss

    @property
    def dead_features(self) -> torch.Tensor:
        if self.total_batches == 0:
            return torch.zeros(self.cfg.d_hidden, dtype=torch.bool)
        avg_freq = self.feature_act_count.float() / self.total_batches.float()
        return avg_freq < 1e-7

    def resample_dead_features(self, mlp_input: torch.Tensor, mlp_output: torch.Tensor):
        """Reinitialize dead features using high-error examples."""
        dead_mask = self.dead_features
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return 0

        with torch.no_grad():
            pred, _, _ = self(mlp_input)
            losses = (mlp_output - pred).pow(2).sum(dim=-1)
            _, top_idx = losses.topk(min(n_dead, len(losses)))

            replacement_in = mlp_input[top_idx]
            replacement_in = replacement_in / replacement_in.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            dead_indices = dead_mask.nonzero(as_tuple=True)[0][:len(top_idx)]
            self.W_enc.data[dead_indices] = replacement_in[:len(dead_indices)] * 0.2
            self.b_enc.data[dead_indices] = 0.0
            self.feature_act_count[dead_indices] = 0

        return n_dead

    def save(self, path: str):
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        state = {
            k: v
            for k, v in self.state_dict().items()
            if not k.startswith("feature_act") and k != "total_batches"
        }
        save_file(state, str(save_dir / "transcoder_weights.safetensors"))

        with open(save_dir / "cfg.json", "w") as f:
            json.dump(vars(self.cfg), f, indent=2)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> Transcoder:
        save_dir = Path(path)
        with open(save_dir / "cfg.json") as f:
            cfg_dict = json.load(f)

        cfg = TranscoderConfig(**cfg_dict)
        tc = cls(cfg)

        weights = load_file(
            str(save_dir / "transcoder_weights.safetensors"), device=device
        )
        tc.load_state_dict(weights, strict=False)
        return tc


def _count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _format_tokens(n: int) -> str:
    """Format a token count into a human-readable string (e.g., 1.5M, 200K)."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _format_duration(seconds: float) -> str:
    """Format seconds into h:mm:ss or m:ss."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


class TranscoderTrainer:
    """
    Training loop for transcoders.

    Supports single-GPU and multi-GPU distributed training via PyTorch DDP.
    When launched with ``torchrun``, distributed mode is auto-detected.
    """

    def __init__(
        self,
        transcoder: Transcoder,
        train_cfg: TrainingConfig,
    ):
        self.transcoder = transcoder
        self.cfg = train_cfg

        # -- Distributed state --
        self._distributed = False
        self._rank = 0
        self._local_rank = 0
        self._world_size = 1

        if torch.distributed.is_initialized():
            self._distributed = True
            self._rank = torch.distributed.get_rank()
            self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self._world_size = torch.distributed.get_world_size()
            self.device = torch.device(f"cuda:{self._local_rank}")
        else:
            self.device = torch.device(train_cfg.device)

        self.transcoder.to(self.device)

        # Wrap in DDP when distributed
        if self._distributed:
            self._ddp_model = torch.nn.parallel.DistributedDataParallel(
                self.transcoder,
                device_ids=[self._local_rank],
                output_device=self._local_rank,
            )
        else:
            self._ddp_model = None

        self.optimizer = torch.optim.Adam(
            self.transcoder.parameters(),
            lr=train_cfg.lr,
            betas=(train_cfg.beta1, train_cfg.beta2),
            weight_decay=train_cfg.weight_decay,
        )

        self._step = 0
        self._tokens_seen = 0
        self._wandb_run = None

    @property
    def _is_rank0(self) -> bool:
        """True when this is the primary process (or non-distributed)."""
        return self._rank == 0

    @property
    def _forward_model(self) -> nn.Module:
        """Return the DDP-wrapped model if distributed, else the raw transcoder."""
        if self._ddp_model is not None:
            return self._ddp_model
        return self.transcoder

    def _init_wandb(self):
        if not self._is_rank0:
            return
        if self.cfg.wandb_project:
            import wandb

            self._wandb_run = wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.wandb_run_name or None,
                config={
                    "transcoder": vars(self.transcoder.cfg),
                    "training": vars(self.cfg),
                },
            )

    def _lr_schedule(self) -> float:
        if self._tokens_seen < self.cfg.warmup_tokens:
            return self._tokens_seen / max(self.cfg.warmup_tokens, 1)
        return 1.0

    def _sync_loss(self, loss_tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce the loss across ranks so logging is accurate.

        Returns the mean loss across all ranks. In non-distributed mode
        this is a no-op that returns the input unchanged.
        """
        if not self._distributed:
            return loss_tensor
        reduced = loss_tensor.detach().clone()
        torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
        reduced /= self._world_size
        return reduced

    def _log_startup_summary(self):
        """Log a comprehensive summary of the training configuration at startup."""
        if not self._is_rank0:
            return

        tc_cfg = self.transcoder.cfg
        tr_cfg = self.cfg
        param_count = _count_parameters(self.transcoder)

        logger.info("=" * 72)
        logger.info("TRANSCODER TRAINING -- CONFIGURATION SUMMARY")
        logger.info("=" * 72)

        # Model architecture
        logger.info("-- Model Architecture --")
        logger.info("  d_in:             %d", tc_cfg.d_in)
        logger.info("  d_out:            %d", tc_cfg.d_out)
        logger.info("  d_hidden:         %d", tc_cfg.d_hidden)
        logger.info("  expansion_factor: %d", tc_cfg.expansion_factor)
        logger.info("  activation_fn:    %s", tc_cfg.activation_fn)
        if tc_cfg.activation_fn == "topk":
            logger.info("  k (topk):         %d", tc_cfg.k)
        elif tc_cfg.activation_fn == "jumprelu":
            logger.info("  jumprelu_thresh:  %.4f", tc_cfg.jumprelu_threshold)
        logger.info("  has_skip:         %s", tc_cfg.has_skip)
        logger.info("  total parameters: %s (%d)", _format_tokens(param_count), param_count)

        # Device info
        logger.info("-- Device --")
        logger.info("  device:           %s", self.device)
        if self.device.type == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(self.device)
            gpu_mem = torch.cuda.get_device_properties(self.device).total_mem / (1024 ** 3)
            logger.info("  GPU:              %s (%.1f GB)", gpu_name, gpu_mem)
        logger.info("  dtype:            %s", tr_cfg.dtype)

        # Distributed info
        if self._distributed:
            logger.info("-- Distributed --")
            logger.info("  world_size:       %d", self._world_size)
            logger.info("  backend:          nccl")

        # Training hyperparameters
        logger.info("-- Training Hyperparameters --")
        logger.info("  lr:               %.2e", tr_cfg.lr)
        logger.info("  betas:            (%.3f, %.4f)", tr_cfg.beta1, tr_cfg.beta2)
        logger.info("  weight_decay:     %.2e", tr_cfg.weight_decay)
        logger.info("  batch_size:       %d", tr_cfg.batch_size)
        logger.info("  total_tokens:     %s", _format_tokens(tr_cfg.total_tokens))
        logger.info("  warmup_tokens:    %s", _format_tokens(tr_cfg.warmup_tokens))

        # Logging and checkpoints
        logger.info("-- Logging & Checkpoints --")
        logger.info("  log_every:        %d steps", tr_cfg.log_every)
        logger.info("  checkpoint_every: %s tokens", _format_tokens(tr_cfg.checkpoint_every))
        logger.info("  checkpoint_dir:   %s", tr_cfg.checkpoint_dir or "(none)")
        logger.info("  wandb_project:    %s", tr_cfg.wandb_project or "(disabled)")

        # Dead feature handling
        logger.info("-- Dead Feature Handling --")
        logger.info("  resample_dead:    %s", tr_cfg.resample_dead)
        if tr_cfg.resample_dead:
            logger.info("  resample_every:   %d steps", tr_cfg.resample_every)
        logger.info("  dead_feat_window: %d", tr_cfg.dead_feature_window)
        logger.info("  dead_feat_thresh: %.1e", tr_cfg.dead_feature_threshold)

        logger.info("=" * 72)

    def _log_periodic_summary(
        self,
        loss: float,
        l0: float,
        cosine_sim: float,
        n_dead: int,
        elapsed: float,
    ):
        """Log a rich periodic summary block to the logger."""
        if not self._is_rank0:
            return

        d_hidden = self.transcoder.cfg.d_hidden
        dead_pct = (n_dead / d_hidden) * 100.0 if d_hidden > 0 else 0.0
        throughput = self._tokens_seen / elapsed if elapsed > 0 else 0.0
        current_lr = self.cfg.lr * self._lr_schedule()

        logger.info("-" * 60)
        logger.info(
            "Step %d | Tokens: %s / %s (%.1f%%)",
            self._step,
            _format_tokens(self._tokens_seen),
            _format_tokens(self.cfg.total_tokens),
            (self._tokens_seen / self.cfg.total_tokens) * 100.0,
        )
        logger.info("  Reconstruction loss : %.6f", loss)
        logger.info("  Cosine similarity   : %.4f", cosine_sim)
        logger.info("  L0 sparsity         : %.2f", l0)
        logger.info(
            "  Dead features       : %d / %d (%.1f%%)",
            n_dead,
            d_hidden,
            dead_pct,
        )
        logger.info("  Learning rate       : %.2e", current_lr)
        logger.info(
            "  Throughput          : %.0f tokens/sec",
            throughput,
        )
        logger.info("  Time elapsed        : %s", _format_duration(elapsed))
        if self._distributed:
            logger.info("  World size          : %d", self._world_size)
        logger.info("-" * 60)

    def _log_final_summary(self, elapsed: float):
        """Log a final summary at the end of training."""
        if not self._is_rank0:
            return

        throughput = self._tokens_seen / elapsed if elapsed > 0 else 0.0

        logger.info("=" * 72)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 72)
        logger.info("  Total steps:      %d", self._step)
        logger.info("  Total tokens:     %s (%d)", _format_tokens(self._tokens_seen), self._tokens_seen)
        logger.info("  Total time:       %s", _format_duration(elapsed))
        logger.info("  Avg throughput:   %.0f tokens/sec", throughput)
        if self._distributed:
            logger.info("  World size:       %d", self._world_size)
        if self.cfg.checkpoint_dir:
            final_path = os.path.join(self.cfg.checkpoint_dir, "transcoder_final")
            logger.info("  Final checkpoint: %s", final_path)
        logger.info("=" * 72)

    def train(self, paired_activation_iter):
        """
        Train the transcoder on paired (mlp_input, mlp_output) batches.

        Args:
            paired_activation_iter: Iterator yielding
                (mlp_input, mlp_output) tensor tuples.
        """
        self._init_wandb()
        self._log_startup_summary()
        self.transcoder.train()

        train_start_time = time.time()
        last_log_time = train_start_time

        if self._is_rank0:
            logger.info("Starting training loop...")

        # Only show tqdm on rank 0
        pbar = tqdm(
            total=self.cfg.total_tokens,
            desc="Training Transcoder",
            unit="tok",
            unit_scale=True,
            smoothing=0.05,
            disable=not self._is_rank0,
        )

        forward_model = self._forward_model

        for mlp_in, mlp_out in paired_activation_iter:
            if self._tokens_seen >= self.cfg.total_tokens:
                break

            mlp_in = mlp_in.to(self.device)
            mlp_out = mlp_out.to(self.device)
            batch_tokens = mlp_in.shape[0]

            lr_mult = self._lr_schedule()
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.cfg.lr * lr_mult

            pred, h, loss = forward_model(mlp_in, mlp_out)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self._step += 1
            self._tokens_seen += batch_tokens

            # Update tqdm with throughput (rank 0 only -- pbar disabled on others)
            now = time.time()
            elapsed = now - train_start_time
            throughput = self._tokens_seen / elapsed if elapsed > 0 else 0.0

            pbar.update(batch_tokens)
            pbar.set_postfix_str(
                f"loss={loss.item():.4f} | {throughput:.0f} tok/s",
                refresh=False,
            )

            if self._step % self.cfg.log_every == 0:
                # Sync loss across ranks for accurate logging
                synced_loss = self._sync_loss(loss)

                with torch.no_grad():
                    l0 = (h > 0).float().sum(dim=-1).mean().item()
                    cosine_sim = F.cosine_similarity(
                        pred, mlp_out, dim=-1
                    ).mean().item()
                    n_dead = self.transcoder.dead_features.sum().item()

                loss_val = synced_loss.item()

                if self._is_rank0:
                    metrics = {
                        "loss": loss_val,
                        "l0": l0,
                        "cosine_sim": cosine_sim,
                        "dead_features": n_dead,
                        "tokens_seen": self._tokens_seen,
                    }

                    if self._wandb_run:
                        import wandb
                        wandb.log(metrics, step=self._step)

                    # Rich tqdm postfix with all key metrics
                    d_hidden = self.transcoder.cfg.d_hidden
                    dead_pct = (n_dead / d_hidden) * 100.0 if d_hidden > 0 else 0.0
                    pbar.set_postfix_str(
                        f"loss={loss_val:.4f} | L0={l0:.1f} | cos={cosine_sim:.4f} "
                        f"| dead={n_dead}({dead_pct:.0f}%) | {throughput:.0f} tok/s",
                        refresh=True,
                    )

                # Periodic rich summary log (rank 0 gated inside the method)
                self._log_periodic_summary(
                    loss=loss_val,
                    l0=l0,
                    cosine_sim=cosine_sim,
                    n_dead=n_dead,
                    elapsed=elapsed,
                )
                last_log_time = now

            if (
                self.cfg.resample_dead
                and self._step % self.cfg.resample_every == 0
                and self._step > 0
            ):
                n_resampled = self.transcoder.resample_dead_features(mlp_in, mlp_out)
                if self._is_rank0:
                    if n_resampled > 0:
                        logger.info(
                            "Resampled %d dead features at step %d (%.1f%% of %d total features)",
                            n_resampled,
                            self._step,
                            (n_resampled / self.transcoder.cfg.d_hidden) * 100.0,
                            self.transcoder.cfg.d_hidden,
                        )
                    else:
                        logger.info(
                            "Dead feature resampling triggered at step %d -- no dead features found",
                            self._step,
                        )
                self.transcoder.feature_act_count.zero_()
                self.transcoder.total_batches.zero_()

            if (
                self.cfg.checkpoint_dir
                and self._tokens_seen % self.cfg.checkpoint_every < batch_tokens
            ):
                if self._is_rank0:
                    ckpt_path = os.path.join(
                        self.cfg.checkpoint_dir,
                        f"transcoder_step{self._step}",
                    )
                    self.transcoder.save(ckpt_path)
                    logger.info(
                        "Checkpoint saved: %s (step %d, %s tokens)",
                        ckpt_path,
                        self._step,
                        _format_tokens(self._tokens_seen),
                    )
                # All ranks wait until rank 0 finishes saving
                if self._distributed:
                    torch.distributed.barrier()

        pbar.close()

        total_elapsed = time.time() - train_start_time

        if self.cfg.checkpoint_dir:
            if self._is_rank0:
                final_path = os.path.join(self.cfg.checkpoint_dir, "transcoder_final")
                self.transcoder.save(final_path)
                logger.info("Final checkpoint saved: %s", final_path)
            if self._distributed:
                torch.distributed.barrier()

        self._log_final_summary(total_elapsed)

        if self._wandb_run:
            import wandb
            wandb.finish()
