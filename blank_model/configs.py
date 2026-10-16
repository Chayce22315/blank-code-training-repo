"""Model/training presets. These are experiments, not capability guarantees."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int = 259  # 256 byte values + BOS + EOS + PAD
    context_length: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_accum_steps: int = 4
    log_every: int = 10
    save_every: int = 100
    eval_every: int = 100
    max_grad_norm: float = 1.0


MODEL_CONFIGS = {
    "blank": ModelConfig(name="blank", d_model=256, n_layers=6, n_heads=8, context_length=256),
    "bigbang": ModelConfig(name="bigbang", d_model=512, n_layers=8, n_heads=8, context_length=512),
    "supernova": ModelConfig(name="supernova", d_model=768, n_layers=12, n_heads=12, context_length=512),
}

TRAIN_CONFIGS = {
    "blank": TrainConfig(batch_size=8, grad_accum_steps=4),
    "bigbang": TrainConfig(batch_size=2, grad_accum_steps=8),
    "supernova": TrainConfig(batch_size=1, grad_accum_steps=16),
}


def parameter_count_hint(name: str) -> int:
    """Return an approximate parameter count without constructing a model."""
    cfg = MODEL_CONFIGS[name]
    # token + position embeddings, attention/MLP blocks, and final projection;
    # output weights are tied to token embeddings in the model.
    d, v, layers = cfg.d_model, cfg.vocab_size, cfg.n_layers
    return v * d + cfg.context_length * d + layers * (4 * d * d + 8 * d * d) + 2 * d + v * d


def as_dict(name: str) -> dict:
    return {**asdict(MODEL_CONFIGS[name]), "train": asdict(TRAIN_CONFIGS[name])}
