# training plan

## phase 0: prove the pipeline

- use the byte tokenizer and small blank configuration.
- validate the model forward pass and gradients.
- use a tiny, explicitly licensed text corpus and check that training loss moves in the right direction.
- save checkpoints outside git and verify generation can reload them.

## phase 1: data and evaluation

- maintain a source-level license/provenance manifest.
- remove duplicates and malformed documents; keep a held-out validation split.
- track training loss, validation loss, tokens processed, wall-clock time, peak VRAM, and exact config/seed.
- create a fixed evaluation set that is not included in training data.

## phase 2: improve blank

- test a trained subword tokenizer against byte-level tokenization.
- add a robust dataset streaming/cache layer.
- run repeated seeds and keep checkpoints only when validation and evaluations improve.
- after base pretraining, design supervised instruction tuning with permitted examples and a clear chat format.

## phase 3: bigbang and supernova

- reuse the tested training pipeline, tokenizer version, dataset manifests, evaluation suite, and logging.
- train each size with its own model config and sufficient compute/data. do not assume blank weights can simply be resized into the larger models.
- compare parameter count, tokens trained, validation loss, task evaluations, inference speed, and memory. larger is a budget, not a guarantee.

## hardware reality

the first blank experiment should be small enough to debug on a consumer gpu. the current 8gb vram / 16gb ram machine is suitable for learning and small experiments, not training a competitive frontier assistant from scratch. supernova may need much more compute and memory, gradient checkpointing, distributed training, or cloud resources. this plan does not depend on paid services.
