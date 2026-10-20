# vta coding dataset: starter pack

## purpose

This folder contains a small, original seed set for future Vibe To App (VTA) instruction tuning. It focuses on app planning, code generation, debugging, testing, adaptive UI, provider setup, security, and truthful reporting.

## files

- `data/instruction/vta_seed.jsonl`: 20 original user/assistant examples in JSON Lines format.
- Each record has `id`, `category`, `source`, and `messages`.
- `source: "original"` means the examples were authored for this repository rather than copied from third-party datasets.

## important limitation

The current `blank_model.train` program is a **plain next-token language-model trainer**. It does not yet perform supervised instruction fine-tuning from this JSONL message schema. Do not feed this JSONL file to the existing text trainer and expect it to understand chat roles. A dedicated SFT data loader/trainer must first convert each record into the model's chat format and mask loss on user messages if that is the chosen training objective.

This 20-example seed set is a format and quality starting point, not enough data to create a capable coding assistant. The from-scratch model configs are very small research experiments. A practical VTA product should use a capable configured provider for immediate use while the local model and training pipeline are developed separately.

## dataset rules

- Keep credentials, personal data, private conversations, and secrets out of the corpus.
- Keep evaluation prompts in a separate held-out file; never train on the test set.
- Track source, license, revision, date retrieved, and preprocessing for every external source.
- Review the current terms for each external dataset before use or redistribution.
- Prefer examples with verifiable outcomes, tests, and clear expected behavior.
- Deduplicate examples and inspect a sample manually before training.

## next steps

1. Add a JSONL schema validator and dataset quality checks.
2. Add a supervised fine-tuning trainer compatible with the model architecture.
3. Expand the original seed set with reviewed examples and carefully licensed code/instruction datasets.
4. Create a separate held-out evaluation set covering generation, debugging, planning, and safety.
