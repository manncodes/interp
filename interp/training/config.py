"""Training configuration dataclasses for SAEs and transcoders."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SAEConfig:
    """Configuration for a single Sparse Autoencoder."""

    d_in: int = 768
    expansion_factor: int = 32
    activation_fn: str = "topk"  # "topk", "jumprelu", "relu"
    k: int = 32  # for topk
    jumprelu_threshold: float = 0.01  # for jumprelu
    normalize_decoder: bool = True
    tied_init: bool = True  # initialize W_dec = W_enc.T

    @property
    def d_sae(self) -> int:
        return self.d_in * self.expansion_factor


@dataclass
class TranscoderConfig:
    """Configuration for a per-layer transcoder or adapter transcoder."""

    d_in: int = 768
    d_out: int = 768
    expansion_factor: int = 32
    activation_fn: str = "topk"
    k: int = 32
    jumprelu_threshold: float = 0.01
    has_skip: bool = True  # skip transcoder (affine skip connection)

    @property
    def d_hidden(self) -> int:
        return self.d_in * self.expansion_factor


@dataclass
class TrainingConfig:
    """Shared training hyperparameters."""

    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    batch_size: int = 4096
    total_tokens: int = 100_000_000  # 100M tokens
    warmup_tokens: int = 1_000_000
    checkpoint_every: int = 10_000_000
    log_every: int = 1000
    l1_coeff: float = 1e-3  # only used for relu activation
    dead_feature_window: int = 5000
    dead_feature_threshold: float = 1e-8
    resample_dead: bool = True
    resample_every: int = 25000
    device: str = "cuda"
    dtype: str = "float32"
    wandb_project: str = ""
    wandb_run_name: str = ""
    checkpoint_dir: str = "./checkpoints"
    seed: int = 42

    # Activation source
    cache_dir: str = ""  # if set, load from cached activations
    dataset_name: str = ""  # HuggingFace dataset name
    dataset_split: str = "train"
    text_column: str = "text"
    context_len: int = 1024
    model_path: str = ""  # path to SplitLlama config dir

    # Hook points to train on (list of hook names)
    hook_names: list[str] = field(default_factory=list)
