"""
nnsight-based wrapper for SplitLlama that provides standardized hook points
for activation extraction, caching, and intervention.

Hook point naming convention:
    embed                           - token embedding output
    layers_first.{i}.resid_post     - residual stream after layer i (8B group)
    layers_first.{i}.attn_out       - attention output at layer i (8B group)
    layers_first.{i}.mlp_in         - MLP input at layer i (8B group)
    layers_first.{i}.mlp_out        - MLP output at layer i (8B group)
    adapter_out                     - adapter output (dimension bridge)
    layers_last.{j}.resid_post      - residual stream after layer j (70B group)
    layers_last.{j}.attn_out        - attention output at layer j (70B group)
    layers_last.{j}.mlp_in          - MLP input at layer j (70B group)
    layers_last.{j}.mlp_out         - MLP output at layer j (70B group)
    norm_out                        - final RMSNorm output
    logits                          - lm_head output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from nnsight import NNsight

from interp.models.split_llama import SplitLlama, SplitLlamaConfig


@dataclass
class HookPoint:
    """Metadata about a single hookable site in the model."""

    name: str
    module_path: str
    d_model: int
    group: str  # "first", "adapter", "last", "embed", "head"
    layer_idx: int | None = None
    site: str = "resid"  # "resid", "attn", "mlp_in", "mlp_out", "embed", "logits"


@dataclass
class HookTopology:
    """Complete description of all hookable sites in the model."""

    d_model_first: int
    d_model_last: int
    n_layers_first: int
    n_layers_last: int
    vocab_size: int
    hook_points: dict[str, HookPoint] = field(default_factory=dict)

    @property
    def all_resid_hooks(self) -> list[str]:
        return [
            name
            for name, hp in self.hook_points.items()
            if hp.site == "resid"
        ]

    @property
    def all_mlp_in_hooks(self) -> list[str]:
        return [
            name
            for name, hp in self.hook_points.items()
            if hp.site == "mlp_in"
        ]

    @property
    def all_mlp_out_hooks(self) -> list[str]:
        return [
            name
            for name, hp in self.hook_points.items()
            if hp.site == "mlp_out"
        ]

    @property
    def first_group_hooks(self) -> list[str]:
        return [
            name
            for name, hp in self.hook_points.items()
            if hp.group == "first"
        ]

    @property
    def last_group_hooks(self) -> list[str]:
        return [
            name
            for name, hp in self.hook_points.items()
            if hp.group == "last"
        ]


class HookedSplitLlama:
    """
    Wraps a SplitLlama model with nnsight for transparent activation access.

    Usage:
        model = SplitLlama(config)
        hooked = HookedSplitLlama(model)

        # Extract activations
        acts = hooked.run_with_cache(
            input_ids,
            hook_names=["layers_first.0.resid_post", "adapter_out"]
        )

        # Get topology for configuring SAE training
        topo = hooked.topology
        print(topo.d_model_first, topo.d_model_last)
    """

    def __init__(self, model: SplitLlama, device: str | torch.device = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self._nnsight = NNsight(self.model)
        self._topology = self._discover_topology()

    @classmethod
    def from_config(
        cls,
        config: SplitLlamaConfig,
        device: str | torch.device = "cpu",
    ) -> HookedSplitLlama:
        model = SplitLlama(config)
        return cls(model, device=device)

    @property
    def topology(self) -> HookTopology:
        return self._topology

    def _discover_topology(self) -> HookTopology:
        """Auto-discover all hookable sites from the model structure."""
        m = self.model
        d_first = m._config_8b.hidden_size
        d_last = m._config_70b.hidden_size
        n_first = len(m.layers_first)
        n_last = len(m.layers_last)

        topo = HookTopology(
            d_model_first=d_first,
            d_model_last=d_last,
            n_layers_first=n_first,
            n_layers_last=n_last,
            vocab_size=m.config.vocab_size,
        )

        # Embedding
        topo.hook_points["embed"] = HookPoint(
            name="embed",
            module_path="embed_tokens",
            d_model=d_first,
            group="embed",
            site="embed",
        )

        # First group (8B layers)
        for i in range(n_first):
            prefix = f"layers_first.{i}"
            topo.hook_points[f"{prefix}.resid_post"] = HookPoint(
                name=f"{prefix}.resid_post",
                module_path=f"layers_first.{i}",
                d_model=d_first,
                group="first",
                layer_idx=i,
                site="resid",
            )
            topo.hook_points[f"{prefix}.attn_out"] = HookPoint(
                name=f"{prefix}.attn_out",
                module_path=f"layers_first.{i}.self_attn",
                d_model=d_first,
                group="first",
                layer_idx=i,
                site="attn",
            )
            topo.hook_points[f"{prefix}.mlp_in"] = HookPoint(
                name=f"{prefix}.mlp_in",
                module_path=f"layers_first.{i}.mlp",
                d_model=d_first,
                group="first",
                layer_idx=i,
                site="mlp_in",
            )
            topo.hook_points[f"{prefix}.mlp_out"] = HookPoint(
                name=f"{prefix}.mlp_out",
                module_path=f"layers_first.{i}.mlp",
                d_model=d_first,
                group="first",
                layer_idx=i,
                site="mlp_out",
            )

        # Adapter
        adapter_path = (
            "adapter_linear_2" if m._use_mlp_adapter else "adapter"
        )
        topo.hook_points["adapter_out"] = HookPoint(
            name="adapter_out",
            module_path=adapter_path,
            d_model=d_last,
            group="adapter",
            site="resid",
        )

        # Last group (70B layers)
        for j in range(n_last):
            prefix = f"layers_last.{j}"
            global_idx = n_first + j
            topo.hook_points[f"{prefix}.resid_post"] = HookPoint(
                name=f"{prefix}.resid_post",
                module_path=f"layers_last.{j}",
                d_model=d_last,
                group="last",
                layer_idx=global_idx,
                site="resid",
            )
            topo.hook_points[f"{prefix}.attn_out"] = HookPoint(
                name=f"{prefix}.attn_out",
                module_path=f"layers_last.{j}.self_attn",
                d_model=d_last,
                group="last",
                layer_idx=global_idx,
                site="attn",
            )
            topo.hook_points[f"{prefix}.mlp_in"] = HookPoint(
                name=f"{prefix}.mlp_in",
                module_path=f"layers_last.{j}.mlp",
                d_model=d_last,
                group="last",
                layer_idx=global_idx,
                site="mlp_in",
            )
            topo.hook_points[f"{prefix}.mlp_out"] = HookPoint(
                name=f"{prefix}.mlp_out",
                module_path=f"layers_last.{j}.mlp",
                d_model=d_last,
                group="last",
                layer_idx=global_idx,
                site="mlp_out",
            )

        # Output
        topo.hook_points["logits"] = HookPoint(
            name="logits",
            module_path="lm_head",
            d_model=m.config.vocab_size,
            group="head",
            site="logits",
        )

        return topo

    def _resolve_module(self, module_path: str) -> Any:
        """Resolve a dot-separated path to an nnsight module proxy."""
        obj = self._nnsight
        for part in module_path.split("."):
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return obj

    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        hook_names: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Run a forward pass and cache activations at specified hook points.

        Args:
            input_ids: Token IDs, shape (batch, seq_len)
            hook_names: Which hooks to cache. If None, caches all residual
                stream hooks.

        Returns:
            Dict mapping hook names to activation tensors.
        """
        if hook_names is None:
            hook_names = self._topology.all_resid_hooks

        input_ids = input_ids.to(self.device)
        saved = {}

        with torch.no_grad(), self._nnsight.trace(input_ids, scan=False, validate=False):
            for name in hook_names:
                hp = self._topology.hook_points[name]
                module = self._resolve_module(hp.module_path)

                if hp.site in ("mlp_in",):
                    proxy = module.input[0][0]
                elif hp.site in ("resid",) and hp.group not in ("adapter",):
                    proxy = module.output[0]
                else:
                    proxy = module.output

                saved[name] = proxy.save()

        return {name: val.value for name, val in saved.items()}

    def run_with_interventions(
        self,
        input_ids: torch.Tensor,
        interventions: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Run a forward pass with activation replacements at specified hooks.

        Args:
            input_ids: Token IDs, shape (batch, seq_len)
            interventions: Dict mapping hook names to replacement tensors.

        Returns:
            Logits tensor.
        """
        input_ids = input_ids.to(self.device)

        with torch.no_grad(), self._nnsight.trace(input_ids, scan=False, validate=False):
            for name, replacement in interventions.items():
                hp = self._topology.hook_points[name]
                module = self._resolve_module(hp.module_path)

                if hp.site in ("mlp_in",):
                    module.input[0][0][:] = replacement.to(self.device)
                elif hp.site in ("resid",) and hp.group not in ("adapter",):
                    module.output[0][:] = replacement.to(self.device)
                else:
                    module.output[:] = replacement.to(self.device)

            logits = self._resolve_module("lm_head").output.save()

        return logits.value

    def get_hook_point(self, name: str) -> HookPoint:
        return self._topology.hook_points[name]

    def get_d_model(self, hook_name: str) -> int:
        return self._topology.hook_points[hook_name].d_model

    def get_mlp_pairs(self) -> list[tuple[str, str]]:
        """Return (mlp_in, mlp_out) hook name pairs for transcoder training."""
        pairs = []
        for name, hp in self._topology.hook_points.items():
            if hp.site == "mlp_in":
                out_name = name.replace("mlp_in", "mlp_out")
                if out_name in self._topology.hook_points:
                    pairs.append((name, out_name))
        return pairs
