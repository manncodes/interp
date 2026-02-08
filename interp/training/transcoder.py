"""
Transcoder model and training loop.

A transcoder maps MLP inputs to MLP outputs through a sparse
intermediate representation, enabling weights-based circuit analysis.

Supports:
- Per-layer transcoders (PLT): d_in == d_out (same hidden dim)
- Adapter transcoders: d_in != d_out (bridges 8B -> 70B dimensions)
- Skip transcoders: adds an affine skip connection for better fidelity
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from interp.training.config import TranscoderConfig, TrainingConfig


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


class TranscoderTrainer:
    """Training loop for transcoders."""

    def __init__(
        self,
        transcoder: Transcoder,
        train_cfg: TrainingConfig,
    ):
        self.transcoder = transcoder
        self.cfg = train_cfg
        self.device = torch.device(train_cfg.device)
        self.transcoder.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.transcoder.parameters(),
            lr=train_cfg.lr,
            betas=(train_cfg.beta1, train_cfg.beta2),
            weight_decay=train_cfg.weight_decay,
        )

        self._step = 0
        self._tokens_seen = 0
        self._wandb_run = None

    def _init_wandb(self):
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

    def train(self, paired_activation_iter):
        """
        Train the transcoder on paired (mlp_input, mlp_output) batches.

        Args:
            paired_activation_iter: Iterator yielding
                (mlp_input, mlp_output) tensor tuples.
        """
        self._init_wandb()
        self.transcoder.train()

        pbar = tqdm(
            total=self.cfg.total_tokens, desc="Training Transcoder", unit="tok"
        )

        for mlp_in, mlp_out in paired_activation_iter:
            if self._tokens_seen >= self.cfg.total_tokens:
                break

            mlp_in = mlp_in.to(self.device)
            mlp_out = mlp_out.to(self.device)
            batch_tokens = mlp_in.shape[0]

            lr_mult = self._lr_schedule()
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.cfg.lr * lr_mult

            pred, h, loss = self.transcoder(mlp_in, mlp_out)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self._step += 1
            self._tokens_seen += batch_tokens
            pbar.update(batch_tokens)

            if self._step % self.cfg.log_every == 0:
                with torch.no_grad():
                    l0 = (h > 0).float().sum(dim=-1).mean().item()
                    cosine_sim = F.cosine_similarity(
                        pred, mlp_out, dim=-1
                    ).mean().item()
                    n_dead = self.transcoder.dead_features.sum().item()

                metrics = {
                    "loss": loss.item(),
                    "l0": l0,
                    "cosine_sim": cosine_sim,
                    "dead_features": n_dead,
                    "tokens_seen": self._tokens_seen,
                }

                if self._wandb_run:
                    import wandb
                    wandb.log(metrics, step=self._step)

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    l0=f"{l0:.1f}",
                    cos=f"{cosine_sim:.4f}",
                    dead=n_dead,
                )

            if (
                self.cfg.resample_dead
                and self._step % self.cfg.resample_every == 0
                and self._step > 0
            ):
                n_resampled = self.transcoder.resample_dead_features(mlp_in, mlp_out)
                if n_resampled > 0:
                    self.transcoder.feature_act_count.zero_()
                    self.transcoder.total_batches.zero_()

            if (
                self.cfg.checkpoint_dir
                and self._tokens_seen % self.cfg.checkpoint_every < batch_tokens
            ):
                ckpt_path = os.path.join(
                    self.cfg.checkpoint_dir,
                    f"transcoder_step{self._step}",
                )
                self.transcoder.save(ckpt_path)

        pbar.close()

        if self.cfg.checkpoint_dir:
            self.transcoder.save(
                os.path.join(self.cfg.checkpoint_dir, "transcoder_final")
            )

        if self._wandb_run:
            import wandb
            wandb.finish()
