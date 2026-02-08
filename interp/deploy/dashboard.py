"""
Generate Neuronpedia-compatible dashboard data for trained SAEs.

Produces batch JSON files containing feature dashboards with:
- Top-activating examples
- Logit effects (promoted/suppressed tokens)
- Activation histograms
- Sparsity statistics

Output format is compatible with the SAEDashboard / Neuronpedia upload API.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from interp.training.sae import SparseAutoencoder


class DashboardGenerator:
    """
    Generates Neuronpedia-compatible feature dashboard data.

    Usage:
        gen = DashboardGenerator(sae, model, tokenizer)
        gen.generate(
            dataset_iter,
            output_dir="./dashboard_data",
            n_prompts=1000,
        )
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        lm_head_weight: torch.Tensor,
        tokenizer,
        device: str = "cuda",
    ):
        """
        Args:
            sae: Trained sparse autoencoder.
            lm_head_weight: The language model head weight matrix (vocab, d_model).
            tokenizer: HuggingFace tokenizer for decoding tokens.
            device: Device for computation.
        """
        self.sae = sae.to(device).eval()
        self.lm_head_weight = lm_head_weight.to(device)
        self.tokenizer = tokenizer
        self.device = device

        self.d_sae = sae.cfg.d_sae
        self.d_in = sae.cfg.d_in

    def generate(
        self,
        activation_iter,
        token_ids_iter,
        output_dir: str,
        n_features_per_batch: int = 64,
        n_top_examples: int = 10,
        n_top_logits: int = 10,
        n_histogram_bins: int = 50,
        sparsity_threshold: float = -5,
    ):
        """
        Generate dashboard data for all features.

        Args:
            activation_iter: Iterator yielding activation tensors (B*T, d_in).
            token_ids_iter: Iterator yielding corresponding token ID tensors.
            output_dir: Directory to write batch JSON files.
            n_features_per_batch: Features per output file.
            n_top_examples: Top-activating examples to store per feature.
            n_top_logits: Top logit effects per feature.
            n_histogram_bins: Bins for activation histogram.
            sparsity_threshold: Log10 threshold below which features are "dead".
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Collect per-feature statistics
        max_acts = torch.zeros(self.d_sae, device=self.device)
        act_counts = torch.zeros(self.d_sae, device=self.device)
        total_tokens = 0
        top_activations: dict[int, list[tuple[float, list[int]]]] = {
            i: [] for i in range(self.d_sae)
        }

        # Pass 1: Collect activation statistics
        for acts, token_ids in tqdm(
            zip(activation_iter, token_ids_iter),
            desc="Collecting feature stats",
        ):
            acts = acts.to(self.device)
            with torch.no_grad():
                features = self.sae.encode(acts)  # (B*T, d_sae)

            batch_max = features.max(dim=0).values
            max_acts = torch.maximum(max_acts, batch_max)
            act_counts += (features > 0).float().sum(dim=0)
            total_tokens += features.shape[0]

            # Track top examples per feature (keep top n_top_examples)
            for feat_idx in range(self.d_sae):
                feat_acts = features[:, feat_idx]
                nonzero_mask = feat_acts > 0
                if nonzero_mask.sum() == 0:
                    continue

                nonzero_acts = feat_acts[nonzero_mask]
                nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0]

                top_k = min(n_top_examples, len(nonzero_acts))
                top_vals, top_local_idx = nonzero_acts.topk(top_k)

                for val, local_idx in zip(top_vals, top_local_idx):
                    global_idx = nonzero_indices[local_idx].item()
                    tok_ids = token_ids[global_idx].tolist() if token_ids.dim() > 1 else [token_ids[global_idx].item()]
                    top_activations[feat_idx].append((val.item(), tok_ids))
                    top_activations[feat_idx].sort(key=lambda x: -x[0])
                    top_activations[feat_idx] = top_activations[feat_idx][:n_top_examples]

        # Compute sparsity
        frac_nonzero = act_counts / max(total_tokens, 1)
        log_sparsity = torch.log10(frac_nonzero.clamp(min=1e-10))

        # Compute logit effects for each feature
        logit_effects = self._compute_logit_effects(n_top_logits)

        # Generate batch files
        skipped_indices = []
        n_batches = math.ceil(self.d_sae / n_features_per_batch)

        for batch_idx in range(n_batches):
            start = batch_idx * n_features_per_batch
            end = min(start + n_features_per_batch, self.d_sae)

            batch_features = []
            for feat_idx in range(start, end):
                if log_sparsity[feat_idx] < sparsity_threshold:
                    skipped_indices.append(feat_idx)
                    continue

                feature_data = {
                    "index": feat_idx,
                    "maxActApprox": max_acts[feat_idx].item(),
                    "frac_nonzero": frac_nonzero[feat_idx].item(),
                    "log_sparsity": log_sparsity[feat_idx].item(),
                    "pos_str": logit_effects[feat_idx]["pos_str"],
                    "pos_values": logit_effects[feat_idx]["pos_values"],
                    "neg_str": logit_effects[feat_idx]["neg_str"],
                    "neg_values": logit_effects[feat_idx]["neg_values"],
                    "top_examples": [
                        {
                            "activation": act,
                            "tokens": toks,
                        }
                        for act, toks in top_activations[feat_idx]
                    ],
                }
                batch_features.append(feature_data)

            if batch_features:
                batch_file = out_path / f"batch-{batch_idx}.json"
                with open(batch_file, "w") as f:
                    json.dump(batch_features, f)

        # Write skipped indices
        with open(out_path / "skipped_indexes.json", "w") as f:
            json.dump(skipped_indices, f)

        # Write metadata
        meta = {
            "d_sae": self.d_sae,
            "d_in": self.d_in,
            "total_tokens": total_tokens,
            "n_active_features": int((log_sparsity >= sparsity_threshold).sum().item()),
            "n_dead_features": len(skipped_indices),
        }
        with open(out_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def _compute_logit_effects(
        self, n_top: int = 10
    ) -> dict[int, dict[str, list]]:
        """Compute top promoted/suppressed tokens for each feature."""
        results = {}

        with torch.no_grad():
            W_dec = self.sae.W_dec  # (d_in, d_sae)
            # logit_effects = lm_head @ W_dec -> (vocab, d_sae)
            logit_effects = self.lm_head_weight @ W_dec

            for feat_idx in range(self.d_sae):
                effects = logit_effects[:, feat_idx]

                # Top positive (promoted)
                pos_vals, pos_idx = effects.topk(n_top)
                pos_str = [
                    self.tokenizer.decode([tid]) for tid in pos_idx.tolist()
                ]
                pos_values = pos_vals.tolist()

                # Top negative (suppressed)
                neg_vals, neg_idx = (-effects).topk(n_top)
                neg_str = [
                    self.tokenizer.decode([tid]) for tid in neg_idx.tolist()
                ]
                neg_values = (-neg_vals).tolist()

                results[feat_idx] = {
                    "pos_str": pos_str,
                    "pos_values": pos_values,
                    "neg_str": neg_str,
                    "neg_values": neg_values,
                }

        return results
