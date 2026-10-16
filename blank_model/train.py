"""Train a randomly initialized model from plain-text files."""
import argparse
import json
import math
import random
import time
from pathlib import Path
import torch
from .configs import MODEL_CONFIGS, TRAIN_CONFIGS
from .data import load_corpus, get_batch
from .model import BlankLanguageModel


def evaluate(model, tokens, batch_size, context_length, device, batches=8):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = get_batch(tokens, batch_size, context_length, device)
            _, loss = model(x, y)
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser(description="train a blank-family language model from scratch")
    parser.add_argument("--config", choices=MODEL_CONFIGS, default="blank")
    parser.add_argument("--data", default="data/train", help="text file or directory of permitted text files")
    parser.add_argument("--validation-data", default="data/validation", help="optional validation directory; skipped if empty")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", default=None, help="checkpoint path to resume model and optimizer state")
    parser.add_argument("--compile", action="store_true", help="try torch.compile (optional; may increase startup/memory)")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be >= 1")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg, train_cfg = MODEL_CONFIGS[args.config], TRAIN_CONFIGS[args.config]
    train_tokens = load_corpus(args.data)
    val_tokens = None
    val_path = Path(args.validation_data)
    if val_path.exists() and any(p.is_file() and p.suffix.lower() in {".txt", ".md", ".rst", ".csv", ".jsonl"} for p in val_path.rglob("*")):
        val_tokens = load_corpus(val_path)

    model = BlankLanguageModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["model_config"]["name"] != args.config:
            raise ValueError("Checkpoint model name does not match --config")
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", 0))
        print(f"resumed from step {start_step}")
    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as exc:
            print(f"torch.compile unavailable, continuing without it: {exc}")
    output = Path(args.output or f"checkpoints/{args.config}")
    output.mkdir(parents=True, exist_ok=True)
    print(f"model={args.config} device={device} parameters={sum(p.numel() for p in model.parameters()):,}")
    print(f"training_bytes={train_tokens.numel()} context={model_cfg.context_length} output={output}")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    for step in range(start_step + 1, start_step + args.steps + 1):
        lr_scale = min(step / 20, 1.0) * 0.5 * (1.0 + math.cos(math.pi * min(step, args.steps) / max(args.steps, 1)))
        for group in optimizer.param_groups:
            group["lr"] = train_cfg.learning_rate * max(lr_scale, 0.05)
        accum_loss = 0.0
        for _ in range(train_cfg.grad_accum_steps):
            x, y = get_batch(train_tokens, train_cfg.batch_size, model_cfg.context_length, device)
            _, loss = model(x, y)
            (loss / train_cfg.grad_accum_steps).backward()
            accum_loss += float(loss.detach().item()) / train_cfg.grad_accum_steps
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % train_cfg.log_every == 0 or step == start_step + 1:
            msg = f"step {step} train_loss={accum_loss:.4f} elapsed={time.time()-started:.1f}s"
            if val_tokens is not None and step % train_cfg.eval_every == 0:
                try:
                    msg += f" val_loss={evaluate(model, val_tokens, train_cfg.batch_size, model_cfg.context_length, device):.4f}"
                except ValueError as exc:
                    msg += f" val_loss=skipped ({exc})"
            print(msg, flush=True)
        if step % train_cfg.save_every == 0 or step == start_step + args.steps:
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            state = {
                "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "model_config": model_cfg.__dict__, "train_config": train_cfg.__dict__,
                "seed": args.seed, "train_tokens": int(train_tokens.numel()),
            }
            temp_path = output / "latest.tmp.pt"
            torch.save(state, temp_path)
            temp_path.replace(output / "latest.pt")
            (output / "run.json").write_text(json.dumps({
                "model": args.config, "step": step, "device": str(device),
                "parameters": sum(p.numel() for p in raw_model.parameters()),
                "train_tokens": int(train_tokens.numel()), "elapsed_seconds": round(time.time()-started, 2),
            }, indent=2), encoding="utf-8")
            print(f"saved {output / 'latest.pt'}", flush=True)
    print("training finished. check validation loss and sample quality before making capability claims.")


if __name__ == "__main__":
    main()
