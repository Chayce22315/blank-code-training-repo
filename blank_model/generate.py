"""Generate a continuation from a trained checkpoint."""
import argparse
import torch
from .configs import ModelConfig
from .model import BlankLanguageModel
from .tokenizer import ByteTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()
    if args.tokens < 0:
        parser.error("--tokens must be >= 0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ModelConfig(**checkpoint["model_config"])
    model = BlankLanguageModel(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = ByteTokenizer()
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        prompt_ids = [32]
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output = model.generate(ids, max_new_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
