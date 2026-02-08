# interp

Mechanistic interpretability framework for custom transformer architectures. End-to-end pipeline from trained model to attribution graphs on Neuronpedia.

## Pipeline

```
Custom Transformer
        |
   [1] Hook activations (nnsight)
        |
   [2] Train SAEs (residual stream decomposition)
        |
   [3] Train Transcoders (MLP replacement with sparse features)
        |
   [4] Build Replacement Model (substitute MLPs, freeze attention)
        |
   [5] Compute Attribution Graph (exact linear decomposition)
        |
   [6] Deploy to Neuronpedia (self-hosted dashboards + graphs)
```

## Install

```bash
uv sync

# with training extras (wandb, bitsandbytes)
uv sync --extra train

# with deployment extras (requests)
uv sync --extra deploy

# everything
uv sync --all-extras
```

## Usage

### 1. Train SAEs

```bash
uv run interp-train-saes \
    --model_path /path/to/model_config \
    --dataset_name HuggingFaceFW/fineweb-edu-sample-10BT \
    --hook_names layers_first.0.resid_post layers_last.0.resid_post \
    --expansion_factor 32 --activation_fn topk --k 32 \
    --total_tokens 100000000 \
    --checkpoint_dir ./checkpoints/saes
```

### 2. Train Transcoders

```bash
uv run interp-train-transcoders \
    --model_path /path/to/model_config \
    --dataset_name HuggingFaceFW/fineweb-edu-sample-10BT \
    --train_all --expansion_factor 32 --activation_fn topk --k 64 \
    --total_tokens 200000000 \
    --checkpoint_dir ./checkpoints/transcoders
```

### 3. Multi-GPU Training

Both trainers support distributed training via `torchrun`. Auto-detected from environment, no flags needed:

```bash
torchrun --nproc_per_node=8 -m interp.scripts.train_saes \
    --model_path /path/to/model_config \
    --dataset_name HuggingFaceFW/fineweb-edu-sample-10BT \
    --hook_names layers_first.0.resid_post \
    --checkpoint_dir ./checkpoints/saes

torchrun --nproc_per_node=8 -m interp.scripts.train_transcoders \
    --model_path /path/to/model_config \
    --dataset_name HuggingFaceFW/fineweb-edu-sample-10BT \
    --train_all --checkpoint_dir ./checkpoints/transcoders
```

### 4. Generate Attribution Graphs

```bash
uv run interp-trace \
    --model_path /path/to/model_config \
    --transcoder_dir ./checkpoints/transcoders \
    --prompt "The capital of France is" \
    --output_dir ./graphs
```

### Python API

```python
from interp import HookedSplitLlama, SparseAutoencoder, Transcoder
from interp.models import SplitLlama, SplitLlamaConfig
from interp.circuits.replacement_model import ReplacementModel
from interp.circuits.attribution import AttributionComputer
from interp.circuits.graph import AttributionGraph

# Wrap model with nnsight hooks
config = SplitLlamaConfig(model_path="./model_dir")
model = SplitLlama(config)
hooked = HookedSplitLlama(model, device="cuda")

# Extract activations
acts = hooked.run_with_cache(input_ids, hook_names=["layers_first.0.resid_post"])

# Inspect hook topology
topo = hooked.topology
print(topo.d_model_first)   # 8B hidden dim
print(topo.d_model_last)    # 70B hidden dim
print(topo.all_resid_hooks) # all residual stream hook points

# Build replacement model from trained transcoders
rm = ReplacementModel(hooked, transcoders)
state = rm.build(input_ids)

# Compute attribution graph
computer = AttributionComputer(rm)
edges = computer.compute_full_graph(state, input_ids)
graph = AttributionGraph.from_edges(edges)
pruned = graph.prune(node_threshold=0.8, edge_threshold=0.98)
pruned.save("graph.json")
```

## Architecture

```
interp/
├── models/          Model definitions
├── wrapper/         nnsight hooks + activation caching
├── training/        SAE and Transcoder models + trainers
├── circuits/        Replacement model, attribution, graph pruning
├── deploy/          Neuronpedia dashboard generation + upload
├── eval/            Reconstruction, sparsity, fidelity metrics
└── scripts/         CLI entry points
```

### Hook Points

```
embed                           token embeddings (d_8b)
layers_first.{i}.resid_post     residual stream, 8B group
layers_first.{i}.attn_out       attention output, 8B group
layers_first.{i}.mlp_in         MLP input, 8B group
layers_first.{i}.mlp_out        MLP output, 8B group
adapter_out                     adapter output (d_8b -> d_70b)
layers_last.{j}.resid_post      residual stream, 70B group
layers_last.{j}.attn_out        attention output, 70B group
layers_last.{j}.mlp_in          MLP input, 70B group
layers_last.{j}.mlp_out         MLP output, 70B group
logits                          lm_head output
```

## Key Design Decisions

| Choice | Rationale |
|--------|-----------|
| **nnsight** for activation access | Works with any PyTorch model, no reimplementation needed |
| **TopK** as default activation | Direct sparsity control, no L1 coefficient tuning |
| **Skip transcoders** by default | Pareto improvement over standard transcoders (lower loss, same interpretability) |
| **Safetensors** for all weights | Safe, fast, compatible with HuggingFace ecosystem |
| **Neuronpedia-compatible** output | Self-host or upload to public Neuronpedia for interactive dashboards |

## References

- [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) -- Anthropic's attribution graph methodology
- [Sparse Feature Circuits](https://arxiv.org/abs/2403.19647) -- Marks et al., feature-level circuit discovery
- [Transcoders Find Interpretable LLM Feature Circuits](https://arxiv.org/abs/2406.11944) -- Dunefsky et al.
- [Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093) -- Gao et al. (OpenAI), TopK SAEs
- [SAELens](https://github.com/jbloomAus/SAELens) -- SAE training library
- [EleutherAI/sparsify](https://github.com/EleutherAI/sparsify) -- SAE/transcoder training
- [circuit-tracer](https://github.com/safety-research/circuit-tracer) -- Open-source circuit tracing
- [Neuronpedia](https://github.com/hijohnnylin/neuronpedia) -- Self-hosted interpretability platform
