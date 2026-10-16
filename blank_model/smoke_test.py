"""Fast shape/gradient test. This does not train language capability."""
import torch
from .configs import MODEL_CONFIGS
from .model import BlankLanguageModel


def main():
    cfg = MODEL_CONFIGS["blank"]
    model = BlankLanguageModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = model(x, y)
    assert logits.shape == (2, 32, cfg.vocab_size), logits.shape
    assert loss is not None and torch.isfinite(loss), loss
    loss.backward()
    assert model.token_embedding.weight.grad is not None
    if torch.cuda.is_available():
        gpu_model = model.to("cuda")
        gpu_x, gpu_y = x.to("cuda"), y.to("cuda")
        gpu_logits, gpu_loss = gpu_model(gpu_x, gpu_y)
        assert gpu_logits.is_cuda and torch.isfinite(gpu_loss)
        print("cpu + cuda smoke tests passed")
    else:
        print("cpu smoke test passed (cuda unavailable)")
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
