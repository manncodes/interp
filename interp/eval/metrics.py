"""
Evaluation metrics for SAEs, transcoders, and replacement models.

Covers:
- Reconstruction quality (MSE, cosine similarity, variance explained)
- Sparsity metrics (L0, dead feature ratio, feature density)
- Replacement model fidelity (KL divergence, cross-entropy increase)
- Feature quality (monosemanticity proxy via activation entropy)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from interp.training.sae import SparseAutoencoder
from interp.training.transcoder import Transcoder


@dataclass
class ReconstructionMetrics:
    mse: float
    cosine_similarity: float
    variance_explained: float
    relative_mse: float  # MSE / variance of input


@dataclass
class SparsityMetrics:
    l0: float  # avg number of active features
    dead_feature_ratio: float
    mean_feature_density: float  # avg fraction of examples activating each feature
    max_feature_density: float


@dataclass
class ReplacementMetrics:
    kl_divergence: float
    cross_entropy_increase: float
    logit_mse: float
    top1_agreement: float  # fraction where argmax matches


@dataclass
class FeatureMetrics:
    n_features: int
    n_active: int
    n_dead: int
    mean_max_activation: float
    activation_entropy: float  # lower = more monosemantic (peakier)


def evaluate_sae(
    sae: SparseAutoencoder,
    activation_iter,
    max_batches: int = 100,
    device: str = "cuda",
) -> tuple[ReconstructionMetrics, SparsityMetrics, FeatureMetrics]:
    """
    Comprehensive evaluation of a trained SAE.

    Args:
        sae: The trained sparse autoencoder.
        activation_iter: Iterator yielding activation batches.
        max_batches: Maximum batches to evaluate.
        device: Computation device.

    Returns:
        Tuple of (ReconstructionMetrics, SparsityMetrics, FeatureMetrics).
    """
    sae = sae.to(device).eval()

    total_mse = 0.0
    total_cos = 0.0
    total_l0 = 0.0
    total_var = 0.0
    total_tokens = 0
    n_batches = 0

    feature_act_sum = torch.zeros(sae.cfg.d_sae, device=device)
    feature_max_act = torch.zeros(sae.cfg.d_sae, device=device)

    with torch.no_grad():
        for batch in activation_iter:
            if n_batches >= max_batches:
                break

            batch = batch.to(device)
            x_hat, h, _ = sae(batch)

            batch_size = batch.shape[0]
            total_tokens += batch_size
            n_batches += 1

            total_mse += F.mse_loss(x_hat, batch, reduction="sum").item()
            total_cos += F.cosine_similarity(x_hat, batch, dim=-1).sum().item()
            total_l0 += (h > 0).float().sum(dim=-1).sum().item()
            total_var += batch.var(dim=0).sum().item() * batch_size

            feature_act_sum += (h > 0).float().sum(dim=0)
            feature_max_act = torch.maximum(feature_max_act, h.max(dim=0).values)

    avg_mse = total_mse / max(total_tokens * sae.cfg.d_in, 1)
    avg_cos = total_cos / max(total_tokens, 1)
    avg_var = total_var / max(total_tokens, 1)
    avg_l0 = total_l0 / max(total_tokens, 1)

    recon = ReconstructionMetrics(
        mse=avg_mse,
        cosine_similarity=avg_cos,
        variance_explained=1.0 - (avg_mse / max(avg_var, 1e-10)),
        relative_mse=avg_mse / max(avg_var, 1e-10),
    )

    feature_density = feature_act_sum / max(total_tokens, 1)
    n_dead = (feature_density < 1e-7).sum().item()
    n_active = sae.cfg.d_sae - n_dead

    sparsity = SparsityMetrics(
        l0=avg_l0,
        dead_feature_ratio=n_dead / sae.cfg.d_sae,
        mean_feature_density=feature_density.mean().item(),
        max_feature_density=feature_density.max().item(),
    )

    # Activation entropy as monosemanticity proxy
    density_nonzero = feature_density[feature_density > 0]
    if len(density_nonzero) > 0:
        p = density_nonzero / density_nonzero.sum()
        entropy = -(p * p.log()).sum().item()
    else:
        entropy = 0.0

    features = FeatureMetrics(
        n_features=sae.cfg.d_sae,
        n_active=n_active,
        n_dead=int(n_dead),
        mean_max_activation=feature_max_act[feature_max_act > 0].mean().item()
        if (feature_max_act > 0).any()
        else 0.0,
        activation_entropy=entropy,
    )

    return recon, sparsity, features


def evaluate_transcoder(
    transcoder: Transcoder,
    paired_activation_iter,
    max_batches: int = 100,
    device: str = "cuda",
) -> tuple[ReconstructionMetrics, SparsityMetrics]:
    """
    Evaluate a trained transcoder.

    Args:
        transcoder: The trained transcoder.
        paired_activation_iter: Iterator yielding (mlp_in, mlp_out) tuples.
        max_batches: Maximum batches.
        device: Computation device.

    Returns:
        Tuple of (ReconstructionMetrics, SparsityMetrics).
    """
    transcoder = transcoder.to(device).eval()

    total_mse = 0.0
    total_cos = 0.0
    total_l0 = 0.0
    total_var = 0.0
    total_tokens = 0
    n_batches = 0

    with torch.no_grad():
        for mlp_in, mlp_out in paired_activation_iter:
            if n_batches >= max_batches:
                break

            mlp_in = mlp_in.to(device)
            mlp_out = mlp_out.to(device)

            pred, h, _ = transcoder(mlp_in, mlp_out)

            batch_size = mlp_in.shape[0]
            total_tokens += batch_size
            n_batches += 1

            total_mse += F.mse_loss(pred, mlp_out, reduction="sum").item()
            total_cos += F.cosine_similarity(pred, mlp_out, dim=-1).sum().item()
            total_l0 += (h > 0).float().sum(dim=-1).sum().item()
            total_var += mlp_out.var(dim=0).sum().item() * batch_size

    d_out = transcoder.cfg.d_out
    avg_mse = total_mse / max(total_tokens * d_out, 1)
    avg_cos = total_cos / max(total_tokens, 1)
    avg_var = total_var / max(total_tokens, 1)
    avg_l0 = total_l0 / max(total_tokens, 1)

    recon = ReconstructionMetrics(
        mse=avg_mse,
        cosine_similarity=avg_cos,
        variance_explained=1.0 - (avg_mse / max(avg_var, 1e-10)),
        relative_mse=avg_mse / max(avg_var, 1e-10),
    )

    sparsity = SparsityMetrics(
        l0=avg_l0,
        dead_feature_ratio=0.0,  # computed at training time
        mean_feature_density=0.0,
        max_feature_density=0.0,
    )

    return recon, sparsity


def evaluate_replacement_fidelity(
    original_logits: torch.Tensor,
    replacement_logits: torch.Tensor,
) -> ReplacementMetrics:
    """
    Compare original model logits with replacement model logits.

    Args:
        original_logits: (batch, seq, vocab) from original model.
        replacement_logits: (batch, seq, vocab) from replacement model.

    Returns:
        ReplacementMetrics with fidelity measurements.
    """
    # Flatten to (B*T, vocab)
    orig = original_logits.reshape(-1, original_logits.shape[-1])
    repl = replacement_logits.reshape(-1, replacement_logits.shape[-1])

    # KL divergence
    orig_probs = F.softmax(orig, dim=-1)
    repl_log_probs = F.log_softmax(repl, dim=-1)
    kl = F.kl_div(repl_log_probs, orig_probs, reduction="batchmean").item()

    # Cross-entropy increase
    orig_ce = F.cross_entropy(
        orig[:-1], orig_probs[1:].argmax(dim=-1), reduction="mean"
    ).item()
    repl_ce = F.cross_entropy(
        repl[:-1], orig_probs[1:].argmax(dim=-1), reduction="mean"
    ).item()

    # Logit MSE
    logit_mse = F.mse_loss(repl, orig).item()

    # Top-1 agreement
    top1_match = (orig.argmax(dim=-1) == repl.argmax(dim=-1)).float().mean().item()

    return ReplacementMetrics(
        kl_divergence=kl,
        cross_entropy_increase=repl_ce - orig_ce,
        logit_mse=logit_mse,
        top1_agreement=top1_match,
    )
