import torch
from blank_model.configs import MODEL_CONFIGS
from blank_model.model import BlankLanguageModel
from blank_model.tokenizer import ByteTokenizer


def test_byte_tokenizer_round_trip():
    tokenizer = ByteTokenizer()
    text = "blank 🚀"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_model_shapes_and_loss():
    cfg = MODEL_CONFIGS["blank"]
    model = BlankLanguageModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)
