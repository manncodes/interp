"""
CLI script for generating attribution graphs from trained transcoders.

Usage:
    interp-trace \
        --model_path /path/to/split_llama_config_dir \
        --transcoder_dir ./checkpoints/transcoders \
        --prompt "The capital of France is" \
        --output_dir ./graphs \
        --top_k 50 \
        --node_threshold 0.8 \
        --edge_threshold 0.98
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from interp.circuits.attribution import AttributionComputer
from interp.circuits.graph import AttributionGraph
from interp.circuits.replacement_model import ReplacementConfig, ReplacementModel
from interp.models.split_llama import SplitLlama, SplitLlamaConfig
from interp.training.transcoder import Transcoder
from interp.wrapper.hooked_model import HookedSplitLlama


def parse_args():
    p = argparse.ArgumentParser(description="Generate attribution graphs")

    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--transcoder_dir", type=str, required=True, help="Dir containing per-layer transcoder checkpoints")
    p.add_argument("--device", type=str, default="cuda")

    # Input
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--target_token", type=str, default="", help="Specific output token to trace")

    # Graph parameters
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--node_threshold", type=float, default=0.8)
    p.add_argument("--edge_threshold", type=float, default=0.98)
    p.add_argument("--max_nodes", type=int, default=200)

    # Output
    p.add_argument("--output_dir", type=str, default="./graphs")
    p.add_argument("--output_name", type=str, default="")

    return p.parse_args()


def main():
    args = parse_args()

    # Load model
    print("Loading model...")
    config = SplitLlamaConfig(model_path=args.model_path)
    model = SplitLlama(config)
    hooked = HookedSplitLlama(model, device=args.device)

    # Load tokenizer
    with open(os.path.join(args.model_path, "config.json")) as f:
        model_cfg = json.load(f)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["path8b"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize prompt
    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(args.device)

    print(f"Prompt: {args.prompt}")
    print(f"Token count: {input_ids.shape[1]}")

    # Load transcoders
    print("Loading transcoders...")
    transcoders = _load_transcoders(args.transcoder_dir, hooked, args.device)
    print(f"Loaded {len(transcoders)} transcoders: {list(transcoders.keys())}")

    if not transcoders:
        print("No transcoders found. Train transcoders first.")
        return

    # Build replacement model
    print("Building replacement model...")
    rm_config = ReplacementConfig(use_error_correction=True)
    replacement_model = ReplacementModel(hooked, transcoders, rm_config)

    # Run forward pass and cache states
    print("Running forward pass...")
    state = replacement_model.build(input_ids)

    # Compute attribution graph
    print("Computing attribution edges...")
    computer = AttributionComputer(replacement_model)
    edges = computer.compute_full_graph(
        state,
        input_ids,
        top_k_per_layer=args.top_k,
        top_k_logits=20,
        top_k_inputs=20,
    )

    print(f"Computed {len(edges)} edges")

    # Build and prune graph
    metadata = {
        "prompt": args.prompt,
        "target_token": args.target_token,
        "model_path": args.model_path,
        "n_transcoders": len(transcoders),
        "transcoder_layers": list(transcoders.keys()),
    }

    graph = AttributionGraph.from_edges(edges, metadata=metadata)
    print(f"Full graph: {graph.n_nodes} nodes, {graph.n_edges} edges")

    pruned = graph.prune(
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        max_nodes=args.max_nodes,
    )
    print(f"Pruned graph: {pruned.n_nodes} nodes, {pruned.n_edges} edges")
    print(pruned.summary())

    # Save
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    name = args.output_name or args.prompt[:50].replace(" ", "_").replace("/", "_")
    full_path = out_path / f"{name}_full.json"
    pruned_path = out_path / f"{name}_pruned.json"

    graph.save(str(full_path))
    pruned.save(str(pruned_path))

    print(f"\nSaved full graph: {full_path}")
    print(f"Saved pruned graph: {pruned_path}")


def _load_transcoders(
    transcoder_dir: str,
    hooked: HookedSplitLlama,
    device: str,
) -> dict[str, Transcoder]:
    """Load all transcoder checkpoints from a directory."""
    tc_dir = Path(transcoder_dir)
    transcoders = {}

    # Look for subdirectories containing transcoder checkpoints
    for subdir in sorted(tc_dir.iterdir()):
        if not subdir.is_dir():
            continue

        # Check for final checkpoint first, then latest step checkpoint
        final_path = subdir / "transcoder_final"
        if final_path.exists():
            load_path = final_path
        else:
            # Find the latest step checkpoint
            step_dirs = sorted(
                [d for d in subdir.iterdir() if d.name.startswith("transcoder_step")],
                key=lambda d: int(d.name.split("step")[1]),
            )
            if step_dirs:
                load_path = step_dirs[-1]
            else:
                continue

        # Reconstruct layer key from directory name
        layer_key = subdir.name.replace("_", ".")

        try:
            tc = Transcoder.load(str(load_path), device=device)
            transcoders[layer_key] = tc
        except Exception as e:
            print(f"Failed to load transcoder at {load_path}: {e}")

    return transcoders


if __name__ == "__main__":
    main()
