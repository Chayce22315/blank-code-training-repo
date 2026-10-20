"""Supervised fine-tuning for prompt/response JSONL examples."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from .configs import MODEL_CONFIGS, TRAIN_CONFIGS
from .model import BlankLanguageModel
from .tokenizer import ByteTokenizer, EOS, PAD


def read_examples(path: str | Path) -> list[dict]:
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"Could not find JSONL file: {file}")
    rows = []
    for line_no, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {file}:{line_no}: {exc}") from exc
        prompt = row.get("prompt")
        response = row.get("response")
        if isinstance(prompt, str) and isinstance(response, str) and prompt.strip() and response.strip():
            rows.append({"prompt": prompt, "response": response, "category": row.get("category", "unknown")})
    if not rows:
        raise ValueError(f"No usable prompt/response examples found in {file}")
    return rows


def encode_example(row: dict, tokenizer: ByteTokenizer, context_length: int, device: torch.device):
    prompt_ids = tokenizer.encode(row["prompt"])
    response_ids = tokenizer.encode(row["response"])
    if not response_ids:
        response_ids = [32]

    # Reserve at least half of the short context for the answer; this model uses
    # byte tokens, so 256 tokens is only about 256 ASCII characters.
    response_ids = response_ids[: max(1, context_length // 2)]
    max_prompt = max(1, context_length + 1 - len(response_ids) - 1)
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[-max_prompt:]

    sequence = prompt_ids + response_ids + [EOS]
    prompt_len = len(prompt_ids)
    inputs = sequence[:-1]
    targets = sequence[1:]
    # The model sees the prompt but is only graded on response tokens and EOS.
    masked_prompt_targets = max(0, prompt_len - 1)
    targets[:masked_prompt_targets] = [-100] * masked_prompt_targets

    pad_len = context_length - len(inputs)
    if pad_len < 0:
        inputs = inputs[:context_length]
        targets = targets[:context_length]
        pad_len = 0
    inputs += [PAD] * pad_len
    targets += [-100] * pad_len
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def batch_examples(rows, tokenizer, context_length, device, batch_size, rng):
    selected = [rows[rng.randrange(len(rows))] for _ in range(batch_size)]
    encoded = [encode_example(row, tokenizer, context_length, device) for row in selected]
    return torch.stack([item[0] for item in encoded]), torch.stack([item[1] for item in encoded])


@torch.no_grad()
def evaluate(model, rows, tokenizer, context_length, device, batch_size):
    if not rows:
        return None
    model.eval()
    loss_sum = 0.0
    batches = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        encoded = [encode_example(row, tokenizer, context_length, device) for row in chunk]
        x = torch.stack([item[0] for item in encoded])
        y = torch.stack([item[1] for item in encoded])
        _, loss = model(x, y)
        if loss is not None and torch.isfinite(loss):
            loss_sum += float(loss.item())
            batches += 1
    model.train()
    return loss_sum / batches if batches else None


def main():
    parser = argparse.ArgumentParser(description="supervised fine-tuning on prompt/response JSONL files")
    parser.add_argument("--config", choices=MODEL_CONFIGS, default="blank")
    parser.add_argument("--data", required=True, help="training JSONL with prompt and response fields")
    parser.add_argument("--validation-data", required=True, help="validation JSONL with prompt and response fields")
    parser.add_argument("--steps", type=int, default=2000, help="number of additional optimizer steps")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", required=True, help="checkpoint whose model weights to continue")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be >= 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")

    random.seed(args.seed)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg, train_cfg = MODEL_CONFIGS[args.config], TRAIN_CONFIGS[args.config]
    train_rows = read_examples(args.data)
    val_rows = read_examples(args.validation_data)

    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    if checkpoint["model_config"]["name"] != args.config:
        raise ValueError("Checkpoint model name does not match --config")
    model = BlankLanguageModel(model_cfg).to(device)
    model.load_state_dict(checkpoint["model"])

    # Use a fresh optimizer because this is a new, response-masked training objective.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=train_cfg.weight_decay
    )
    start_step = int(checkpoint.get("step", 0))
    final_step = start_step + args.steps
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteTokenizer()

    print(f"model={args.config} device={device} parameters={sum(p.numel() for p in model.parameters()):,}")
    print(f"resumed model weights from step {start_step}; optimizer reset for supervised fine-tuning")
    print(f"train_examples={len(train_rows)} validation_examples={len(val_rows)}")
    print(f"context={model_cfg.context_length} learning_rate={args.learning_rate} output={output}")

    model.train()
    started = time.time()
    for step in range(start_step + 1, final_step + 1):
        local_step = step - start_step
        warmup = min(local_step / 30, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * local_step / max(args.steps, 1)))
        lr_scale = max(warmup * cosine, 0.1)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * lr_scale

        accum_loss = 0.0
        for _ in range(train_cfg.grad_accum_steps):
            x, y = batch_examples(
                train_rows, tokenizer, model_cfg.context_length, device,
                train_cfg.batch_size, rng
            )
            _, loss = model(x, y)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {step}")
            (loss / train_cfg.grad_accum_steps).backward()
            accum_loss += float(loss.detach().item()) / train_cfg.grad_accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % train_cfg.log_every == 0 or step == start_step + 1:
            message = f"step {step} train_loss={accum_loss:.4f} elapsed={time.time() - started:.1f}s"
            if step % train_cfg.eval_every == 0 or step == final_step:
                val_loss = evaluate(model, val_rows, tokenizer, model_cfg.context_length, device, train_cfg.batch_size)
                if val_loss is not None:
                    message += f" val_loss={val_loss:.4f}"
            print(message, flush=True)

        if step % train_cfg.save_every == 0 or step == final_step:
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "model_config": model_cfg.__dict__,
                "train_config": {**train_cfg.__dict__, "learning_rate": args.learning_rate, "objective": "response-masked-sft"},
                "seed": args.seed,
                "train_examples": len(train_rows),
                "validation_examples": len(val_rows),
            }
            temp_path = output / "latest.tmp.pt"
            torch.save(state, temp_path)
            temp_path.replace(output / "latest.pt")
            (output / "run.json").write_text(json.dumps({
                "model": args.config,
                "step": step,
                "device": str(device),
                "parameters": sum(p.numel() for p in model.parameters()),
                "training_examples": len(train_rows),
                "validation_examples": len(val_rows),
                "elapsed_seconds": round(time.time() - started, 2),
                "objective": "response-masked supervised fine-tuning",
                "parent_checkpoint": str(args.resume),
                "learning_rate": args.learning_rate,
            }, indent=2) + "\n", encoding="utf-8")
            print(f"saved {output / 'latest.pt'}", flush=True)

    print("supervised fine-tuning finished. Test generated answers and task accuracy before claiming capability.")


if __name__ == "__main__":
    main()
