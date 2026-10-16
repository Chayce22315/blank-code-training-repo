"""Load a small plain-text corpus and sample next-token training windows."""
from pathlib import Path
import torch
from .tokenizer import ByteTokenizer

EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".jsonl"}


def load_corpus(path: str | Path) -> torch.Tensor:
    root = Path(path)
    if root.is_file():
        files = [root]
    elif root.exists():
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    else:
        files = []
    if not files:
        raise FileNotFoundError(f"No supported text files found in {root}. Add licensed .txt files first.")
    tokenizer = ByteTokenizer()
    chunks = []
    for file in files:
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"warning: skipping {file}: {exc}")
            continue
        if text.strip():
            chunks.extend(tokenizer.encode(text + "\n\n"))
    if len(chunks) < 2:
        raise ValueError("Corpus must contain at least two bytes of non-empty text.")
    return torch.tensor(chunks, dtype=torch.long)


def get_batch(tokens: torch.Tensor, batch_size: int, context_length: int, device: torch.device):
    if tokens.numel() <= context_length:
        raise ValueError(f"Corpus has {tokens.numel()} tokens but context length is {context_length}; add more text or use a smaller context.")
    starts = torch.randint(0, tokens.numel() - context_length - 1, (batch_size,))
    x = torch.stack([tokens[i:i + context_length] for i in starts])
    y = torch.stack([tokens[i + 1:i + context_length + 1] for i in starts])
    return x.to(device), y.to(device)
