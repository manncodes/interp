"""
Attribution edge computation for circuit tracing.

Computes the causal influence between features in the replacement model
using gradient-based linear attribution. When the replacement model is
locally linear (frozen attention + frozen nonlinearities), edge weights
are exact decompositions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from interp.circuits.replacement_model import (
    CachedForwardState,
    FeatureNode,
    ReplacementModel,
)
from interp.training.transcoder import Transcoder


class AttributionEdge:
    """A directed edge between two nodes in the attribution graph."""

    __slots__ = ("source", "target", "weight", "token_pos")

    def __init__(
        self,
        source: str,
        target: str,
        weight: float,
        token_pos: int | None = None,
    ):
        self.source = source
        self.target = target
        self.weight = weight
        self.token_pos = token_pos

    def __repr__(self) -> str:
        return f"Edge({self.source} -> {self.target}, w={self.weight:.4f})"


class AttributionComputer:
    """
    Computes attribution edges between features in the replacement model.

    For a transcoder with encoder W_enc and decoder W_dec, the virtual weight
    from source feature s to target feature t (across layers) is:

        w_{s->t} = W_dec[:, s] @ M @ W_enc[t, :]^T

    where M captures the linear path between layers (through attention OV
    circuits and residual connections).

    The attribution edge weight is then:
        A_{s->t} = activation_s * w_{s->t}

    When within the same layer group and connected only by residual stream,
    M = I (identity), simplifying to:
        w_{s->t} = W_dec[:, s] @ W_enc[t, :]^T
    """

    def __init__(self, replacement_model: ReplacementModel):
        self.rm = replacement_model

    def compute_direct_feature_effects(
        self,
        state: CachedForwardState,
        source_layer: str,
        target_layer: str,
        top_k: int = 50,
    ) -> list[AttributionEdge]:
        """
        Compute direct attribution edges between features in two layers.

        For adjacent layers connected by residual stream, this computes
        exact virtual weights via decoder-encoder dot products.

        Args:
            state: Cached forward state from replacement model.
            source_layer: Layer key for source features.
            target_layer: Layer key for target features.
            top_k: Return only the top-k strongest edges.

        Returns:
            List of AttributionEdge objects sorted by absolute weight.
        """
        if source_layer not in state.transcoder_outputs:
            return []
        if target_layer not in state.transcoder_outputs:
            return []

        source_tc = self.rm.transcoders[source_layer]
        target_tc = self.rm.transcoders[target_layer]

        source_data = state.transcoder_outputs[source_layer]
        target_data = state.transcoder_outputs[target_layer]

        source_features = source_data["features"]  # (B, T, d_hidden_src)
        target_features = target_data["features"]  # (B, T, d_hidden_tgt)

        # Find active features
        src_active = (source_features > 0).any(dim=0).any(dim=0)
        tgt_active = (target_features > 0).any(dim=0).any(dim=0)

        src_indices = src_active.nonzero(as_tuple=True)[0]
        tgt_indices = tgt_active.nonzero(as_tuple=True)[0]

        if len(src_indices) == 0 or len(tgt_indices) == 0:
            return []

        # Virtual weights: W_dec[:, src] @ W_enc[tgt, :]^T
        # W_dec is (d_out, d_hidden), W_enc is (d_hidden, d_in)
        # For residual connection: d_out of source == d_in of target
        src_decoder_cols = source_tc.W_dec[:, src_indices]  # (d_out, n_src)
        tgt_encoder_rows = target_tc.W_enc[tgt_indices, :]  # (n_tgt, d_in)

        # Check dimension compatibility
        if src_decoder_cols.shape[0] != tgt_encoder_rows.shape[1]:
            # Cross-group connection (different hidden dims)
            # Must trace through the adapter
            return self._compute_cross_group_effects(
                state, source_layer, target_layer,
                src_indices, tgt_indices,
                source_features, target_features,
                source_tc, target_tc,
            )

        # virtual_weights[i, j] = W_dec[:, src_i] @ W_enc[tgt_j, :]^T
        virtual_weights = src_decoder_cols.T @ tgt_encoder_rows.T  # (n_src, n_tgt)

        # Compute edge weights: activation * virtual_weight
        # Use mean activation across batch and positions
        src_mean_acts = source_features[:, :, src_indices].mean(dim=(0, 1))  # (n_src,)

        # edge_weights[i, j] = src_mean_acts[i] * virtual_weights[i, j]
        edge_weights = src_mean_acts.unsqueeze(1) * virtual_weights  # (n_src, n_tgt)

        # Flatten and find top-k
        flat_weights = edge_weights.reshape(-1)
        k = min(top_k, len(flat_weights))
        top_vals, top_flat_idx = flat_weights.abs().topk(k)

        edges = []
        for flat_idx, val in zip(top_flat_idx, top_vals):
            i = flat_idx // edge_weights.shape[1]
            j = flat_idx % edge_weights.shape[1]
            src_feat = src_indices[i].item()
            tgt_feat = tgt_indices[j].item()
            w = edge_weights[i, j].item()

            edges.append(
                AttributionEdge(
                    source=f"{source_layer}/f{src_feat}",
                    target=f"{target_layer}/f{tgt_feat}",
                    weight=w,
                )
            )

        return edges

    def _compute_cross_group_effects(
        self,
        state: CachedForwardState,
        source_layer: str,
        target_layer: str,
        src_indices: torch.Tensor,
        tgt_indices: torch.Tensor,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_tc: Transcoder,
        target_tc: Transcoder,
    ) -> list[AttributionEdge]:
        """
        Compute attribution across the adapter boundary.

        The path is: source decoder -> adapter linear -> target encoder.
        For a linear adapter W_adapt:
            virtual_weight = W_dec[:, s]^T @ W_adapt^T @ W_enc[t, :]^T
        """
        model = self.rm.hooked_model.model
        if model._use_mlp_adapter:
            W_adapt = model.adapter_linear_2.weight @ model.adapter_linear_1.weight
        else:
            W_adapt = model.adapter.weight  # (d_70b, d_8b)

        src_decoder_cols = source_tc.W_dec[:, src_indices]  # (d_8b, n_src)
        tgt_encoder_rows = target_tc.W_enc[tgt_indices, :]  # (n_tgt, d_70b)

        # Path: src_dec -> W_adapt -> tgt_enc
        # virtual_weights = src_dec^T @ W_adapt^T @ tgt_enc^T
        adapted = W_adapt @ src_decoder_cols  # (d_70b, n_src)
        virtual_weights = adapted.T @ tgt_encoder_rows.T  # (n_src, n_tgt)

        src_mean_acts = source_features[:, :, src_indices].mean(dim=(0, 1))
        edge_weights = src_mean_acts.unsqueeze(1) * virtual_weights

        flat_weights = edge_weights.reshape(-1)
        k = min(50, len(flat_weights))
        top_vals, top_flat_idx = flat_weights.abs().topk(k)

        edges = []
        for flat_idx, val in zip(top_flat_idx, top_vals):
            i = flat_idx // edge_weights.shape[1]
            j = flat_idx % edge_weights.shape[1]
            w = edge_weights[i, j].item()
            edges.append(
                AttributionEdge(
                    source=f"{source_layer}/f{src_indices[i].item()}",
                    target=f"{target_layer}/f{tgt_indices[j].item()}",
                    weight=w,
                )
            )

        return edges

    def compute_feature_to_logit_effects(
        self,
        state: CachedForwardState,
        layer_key: str,
        top_k: int = 20,
    ) -> list[AttributionEdge]:
        """
        Compute the direct effect of features on output logits.

        For features in the last layer group, the path is:
            feature decoder -> LayerNorm -> lm_head

        Args:
            state: Cached forward state.
            layer_key: Which layer's features to analyze.
            top_k: Number of top token effects per feature.

        Returns:
            List of edges from features to logit tokens.
        """
        if layer_key not in state.transcoder_outputs:
            return []

        tc = self.rm.transcoders[layer_key]
        tc_data = state.transcoder_outputs[layer_key]
        features = tc_data["features"]

        lm_head_weight = self.rm.hooked_model.model.lm_head.weight  # (vocab, d_model)

        active = (features > 0).any(dim=0).any(dim=0)
        active_idx = active.nonzero(as_tuple=True)[0]

        edges = []
        for feat_idx in active_idx:
            # Decoder column for this feature
            dec_col = tc.W_dec[:, feat_idx]  # (d_model,)
            # Logit effect = lm_head @ decoder_column
            logit_effects = lm_head_weight @ dec_col  # (vocab,)

            top_vals, top_tokens = logit_effects.abs().topk(top_k)
            for token_id, val in zip(top_tokens, top_vals):
                effect = logit_effects[token_id].item()
                edges.append(
                    AttributionEdge(
                        source=f"{layer_key}/f{feat_idx.item()}",
                        target=f"logit/{token_id.item()}",
                        weight=effect,
                    )
                )

        return edges

    def compute_input_to_feature_effects(
        self,
        state: CachedForwardState,
        layer_key: str,
        input_ids: torch.Tensor,
        top_k: int = 20,
    ) -> list[AttributionEdge]:
        """
        Compute attribution from input token embeddings to features.

        For the first layer, the path is:
            token_embedding -> encoder

        Args:
            state: Cached forward state.
            layer_key: Which layer's features to analyze (typically first layer).
            input_ids: Token IDs for labeling.
            top_k: Number of top edges.

        Returns:
            List of edges from input tokens to features.
        """
        if layer_key not in state.transcoder_outputs:
            return []

        tc = self.rm.transcoders[layer_key]
        tc_data = state.transcoder_outputs[layer_key]
        features = tc_data["features"]  # (B, T, d_hidden)
        mlp_input = tc_data["mlp_input"]  # (B, T, d_in)

        active = (features > 0).any(dim=0).any(dim=0)
        active_idx = active.nonzero(as_tuple=True)[0]

        edges = []
        for feat_idx in active_idx:
            enc_row = tc.W_enc[feat_idx, :]  # (d_in,)
            # For each position, the contribution is input @ enc_row
            contributions = mlp_input[0] @ enc_row  # (T,)

            top_vals, top_pos = contributions.abs().topk(min(top_k, len(contributions)))
            for pos, val in zip(top_pos, top_vals):
                token_id = input_ids[0, pos].item()
                edges.append(
                    AttributionEdge(
                        source=f"token/{pos.item()}/{token_id}",
                        target=f"{layer_key}/f{feat_idx.item()}",
                        weight=contributions[pos].item(),
                        token_pos=pos.item(),
                    )
                )

        return edges

    def compute_full_graph(
        self,
        state: CachedForwardState,
        input_ids: torch.Tensor,
        top_k_per_layer: int = 50,
        top_k_logits: int = 20,
        top_k_inputs: int = 20,
    ) -> list[AttributionEdge]:
        """
        Compute the complete attribution graph across all layers.

        Returns all edges: input->features, feature->feature, feature->logits.
        """
        all_edges = []
        layer_keys = sorted(state.transcoder_outputs.keys())

        # Input -> first layer features
        if layer_keys:
            input_edges = self.compute_input_to_feature_effects(
                state, layer_keys[0], input_ids, top_k=top_k_inputs
            )
            all_edges.extend(input_edges)

        # Feature -> feature (between consecutive layers)
        for i in range(len(layer_keys) - 1):
            inter_edges = self.compute_direct_feature_effects(
                state, layer_keys[i], layer_keys[i + 1], top_k=top_k_per_layer
            )
            all_edges.extend(inter_edges)

        # Last layer features -> logits
        if layer_keys:
            logit_edges = self.compute_feature_to_logit_effects(
                state, layer_keys[-1], top_k=top_k_logits
            )
            all_edges.extend(logit_edges)

        return all_edges
