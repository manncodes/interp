"""
Sparse Autoencoder model and training loop.

Supports TopK, JumpReLU, and standard ReLU activation functions.
Includes dead feature resampling and decoder normalization.
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

from interp.training.config import SAEConfig, TrainingConfig


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


class SAETrainer:
    """Training loop for sparse autoencoders."""

    def __init__(
        self,
        sae: SparseAutoencoder,
        train_cfg: TrainingConfig,
    ):
        self.sae = sae
        self.cfg = train_cfg
        self.device = torch.device(train_cfg.device)
        self.sae.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.sae.parameters(),
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
                    "sae": vars(self.sae.cfg),
                    "training": vars(self.cfg),
                },
            )

    def _lr_schedule(self) -> float:
        """Linear warmup then constant."""
        if self._tokens_seen < self.cfg.warmup_tokens:
            return self._tokens_seen / max(self.cfg.warmup_tokens, 1)
        return 1.0

    def train(self, activation_iter):
        """
        Train the SAE on an iterator of activation batches.

        Args:
            activation_iter: Iterator yielding tensors of shape (batch, d_in).
                Can be from ActivationStore.stream() or
                ActivationStore.get_cached_loader().
        """
        self._init_wandb()
        self.sae.train()

        pbar = tqdm(total=self.cfg.total_tokens, desc="Training SAE", unit="tok")

        for batch in activation_iter:
            if self._tokens_seen >= self.cfg.total_tokens:
                break

            batch = batch.to(self.device)
            batch_tokens = batch.shape[0]

            # LR warmup
            lr_mult = self._lr_schedule()
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.cfg.lr * lr_mult

            # Forward
            x_hat, h, loss = self.sae(batch)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Normalize decoder columns
            if self.sae.cfg.normalize_decoder:
                self.sae._normalize_decoder()

            self._step += 1
            self._tokens_seen += batch_tokens
            pbar.update(batch_tokens)

            # Logging
            if self._step % self.cfg.log_every == 0:
                with torch.no_grad():
                    l0 = (h > 0).float().sum(dim=-1).mean().item()
                    recon_loss = F.mse_loss(x_hat, batch).item()
                    cosine_sim = F.cosine_similarity(x_hat, batch, dim=-1).mean().item()
                    n_dead = self.sae.dead_features.sum().item()

                metrics = {
                    "loss": loss.item(),
                    "recon_loss": recon_loss,
                    "l0": l0,
                    "cosine_sim": cosine_sim,
                    "dead_features": n_dead,
                    "lr": self.cfg.lr * lr_mult,
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

            # Resample dead features
            if (
                self.cfg.resample_dead
                and self._step % self.cfg.resample_every == 0
                and self._step > 0
            ):
                n_resampled = self.sae.resample_dead_features(batch)
                if n_resampled > 0:
                    self.sae.feature_act_count.zero_()
                    self.sae.total_batches.zero_()

            # Checkpoint
            if (
                self.cfg.checkpoint_dir
                and self._tokens_seen % self.cfg.checkpoint_every < batch_tokens
            ):
                ckpt_path = os.path.join(
                    self.cfg.checkpoint_dir,
                    f"sae_step{self._step}",
                )
                self.sae.save(ckpt_path)

        pbar.close()

        # Final save
        if self.cfg.checkpoint_dir:
            self.sae.save(os.path.join(self.cfg.checkpoint_dir, "sae_final"))

        if self._wandb_run:
            import wandb
            wandb.finish()
