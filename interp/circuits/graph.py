"""
Attribution graph data structure, pruning, and export.

Builds a directed graph from attribution edges, provides pruning
algorithms to reduce to human-readable size, and exports to JSON
format compatible with Neuronpedia's graph viewer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from interp.circuits.attribution import AttributionEdge


@dataclass
class GraphNode:
    """A node in the attribution graph."""

    id: str
    node_type: str  # "feature", "token", "logit", "error"
    layer: str = ""
    feature_idx: int = -1
    token_id: int = -1
    token_pos: int = -1
    activation: float = 0.0
    label: str = ""
    incoming_weight: float = 0.0
    outgoing_weight: float = 0.0


@dataclass
class AttributionGraph:
    """
    Complete attribution graph with nodes and edges.

    Supports pruning, statistics, and export to JSON.
    """

    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[AttributionEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_edges(
        cls,
        edges: list[AttributionEdge],
        metadata: dict[str, Any] | None = None,
    ) -> AttributionGraph:
        """Build a graph from a list of attribution edges."""
        graph = cls(metadata=metadata or {})

        for edge in edges:
            graph.edges.append(edge)

            # Auto-create nodes
            if edge.source not in graph.nodes:
                graph.nodes[edge.source] = _parse_node_id(edge.source)
            if edge.target not in graph.nodes:
                graph.nodes[edge.target] = _parse_node_id(edge.target)

            graph.nodes[edge.source].outgoing_weight += abs(edge.weight)
            graph.nodes[edge.target].incoming_weight += abs(edge.weight)

        return graph

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def feature_nodes(self) -> list[GraphNode]:
        return [n for n in self.nodes.values() if n.node_type == "feature"]

    @property
    def token_nodes(self) -> list[GraphNode]:
        return [n for n in self.nodes.values() if n.node_type == "token"]

    @property
    def logit_nodes(self) -> list[GraphNode]:
        return [n for n in self.nodes.values() if n.node_type == "logit"]

    def prune(
        self,
        node_threshold: float = 0.8,
        edge_threshold: float = 0.98,
        min_nodes: int = 5,
        max_nodes: int = 200,
    ) -> AttributionGraph:
        """
        Prune the graph to retain only the most important paths.

        Algorithm:
        1. Start from logit nodes (sinks)
        2. Trace backward, accumulating attribution mass
        3. Keep nodes that collectively explain `node_threshold` fraction
           of total attribution at each node
        4. Remove edges below `edge_threshold` percentile

        Args:
            node_threshold: Fraction of attribution to preserve at each node.
            edge_threshold: Fraction of edges to prune (by abs weight).
            min_nodes: Minimum nodes to keep.
            max_nodes: Maximum nodes to keep.

        Returns:
            A new pruned AttributionGraph.
        """
        if not self.edges:
            return AttributionGraph(metadata=self.metadata)

        # Step 1: Edge pruning by absolute weight
        sorted_edges = sorted(self.edges, key=lambda e: abs(e.weight), reverse=True)
        n_keep = max(
            int(len(sorted_edges) * (1 - edge_threshold)),
            min_nodes,
        )
        n_keep = min(n_keep, len(sorted_edges))
        kept_edges = sorted_edges[:n_keep]

        # Step 2: Build adjacency for backward trace
        incoming: dict[str, list[AttributionEdge]] = {}
        for edge in kept_edges:
            incoming.setdefault(edge.target, []).append(edge)

        # Step 3: Backward trace from logit nodes
        logit_ids = [
            nid for nid, n in self.nodes.items() if n.node_type == "logit"
        ]
        if not logit_ids:
            # Fallback: use nodes with no outgoing edges
            targets_set = {e.target for e in kept_edges}
            sources_set = {e.source for e in kept_edges}
            logit_ids = list(targets_set - sources_set)

        visited = set()
        queue = list(logit_ids)

        while queue and len(visited) < max_nodes:
            node_id = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)

            if node_id in incoming:
                # Sort incoming by weight, keep top fraction
                in_edges = sorted(
                    incoming[node_id], key=lambda e: abs(e.weight), reverse=True
                )
                total_weight = sum(abs(e.weight) for e in in_edges)
                if total_weight == 0:
                    continue

                cumulative = 0.0
                for edge in in_edges:
                    if cumulative / total_weight < node_threshold:
                        cumulative += abs(edge.weight)
                        if edge.source not in visited:
                            queue.append(edge.source)

        # Step 4: Filter edges to only those between kept nodes
        final_edges = [
            e for e in kept_edges
            if e.source in visited and e.target in visited
        ]

        # Build pruned graph
        pruned = AttributionGraph.from_edges(final_edges, metadata=self.metadata)
        pruned.metadata["pruned"] = True
        pruned.metadata["original_nodes"] = self.n_nodes
        pruned.metadata["original_edges"] = self.n_edges

        return pruned

    def to_json(self) -> dict:
        """Export to JSON-serializable dict (Neuronpedia-compatible format)."""
        nodes_json = []
        for nid, node in self.nodes.items():
            nodes_json.append({
                "id": nid,
                "type": node.node_type,
                "layer": node.layer,
                "feature_idx": node.feature_idx,
                "token_id": node.token_id,
                "token_pos": node.token_pos,
                "activation": node.activation,
                "label": node.label,
                "incoming_weight": node.incoming_weight,
                "outgoing_weight": node.outgoing_weight,
            })

        edges_json = []
        for edge in self.edges:
            edges_json.append({
                "source": edge.source,
                "target": edge.target,
                "weight": edge.weight,
            })

        return {
            "nodes": nodes_json,
            "edges": edges_json,
            "metadata": self.metadata,
            "stats": {
                "n_nodes": self.n_nodes,
                "n_edges": self.n_edges,
                "n_features": len(self.feature_nodes),
                "n_tokens": len(self.token_nodes),
                "n_logits": len(self.logit_nodes),
            },
        }

    def save(self, path: str):
        """Save graph to JSON file."""
        fpath = Path(path)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(fpath, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> AttributionGraph:
        """Load graph from JSON file."""
        with open(path) as f:
            data = json.load(f)

        edges = [
            AttributionEdge(
                source=e["source"],
                target=e["target"],
                weight=e["weight"],
            )
            for e in data["edges"]
        ]

        graph = cls.from_edges(edges, metadata=data.get("metadata", {}))

        # Restore node metadata
        for node_data in data["nodes"]:
            nid = node_data["id"]
            if nid in graph.nodes:
                graph.nodes[nid].label = node_data.get("label", "")
                graph.nodes[nid].activation = node_data.get("activation", 0.0)

        return graph

    def summary(self) -> str:
        """Print a human-readable summary of the graph."""
        lines = [
            f"Attribution Graph: {self.n_nodes} nodes, {self.n_edges} edges",
            f"  Features: {len(self.feature_nodes)}",
            f"  Tokens: {len(self.token_nodes)}",
            f"  Logits: {len(self.logit_nodes)}",
        ]

        if self.edges:
            weights = [abs(e.weight) for e in self.edges]
            lines.append(
                f"  Edge weights: min={min(weights):.4f}, "
                f"max={max(weights):.4f}, "
                f"mean={sum(weights)/len(weights):.4f}"
            )

        # Top features by outgoing weight
        top_features = sorted(
            self.feature_nodes,
            key=lambda n: n.outgoing_weight,
            reverse=True,
        )[:10]
        if top_features:
            lines.append("  Top features by influence:")
            for f in top_features:
                lines.append(
                    f"    {f.id}: out={f.outgoing_weight:.4f} "
                    f"in={f.incoming_weight:.4f}"
                )

        return "\n".join(lines)


def _parse_node_id(node_id: str) -> GraphNode:
    """Parse a node ID string into a GraphNode."""
    parts = node_id.split("/")

    if parts[0] == "token":
        return GraphNode(
            id=node_id,
            node_type="token",
            token_pos=int(parts[1]) if len(parts) > 1 else -1,
            token_id=int(parts[2]) if len(parts) > 2 else -1,
        )
    elif parts[0] == "logit":
        return GraphNode(
            id=node_id,
            node_type="logit",
            token_id=int(parts[1]) if len(parts) > 1 else -1,
        )
    elif parts[0] == "error":
        return GraphNode(
            id=node_id,
            node_type="error",
            layer=parts[1] if len(parts) > 1 else "",
        )
    else:
        # Feature node: "layers_first.0/f123"
        layer = parts[0]
        feat_idx = int(parts[1][1:]) if len(parts) > 1 and parts[1].startswith("f") else -1
        return GraphNode(
            id=node_id,
            node_type="feature",
            layer=layer,
            feature_idx=feat_idx,
        )
