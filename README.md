# blank model family

training code for the **blank**, **bigbang**, and **supernova** model family. this repository is separate from the blank code app.

> status: early research prototype. the model starts with randomly initialized weights and is trained from scratch. it is not a capable general-purpose assistant yet.

## the family

| model | role | default size target | local training |
|---|---|---:|---|
| blank | everyday use | small experiment, ~5m parameters | intended for the first local run |
| bigbang | regular use | medium experiment, ~25m parameters | may fit in 8gb vram with conservative settings, but training time varies |
| supernova | largest and smartest | larger experiment, ~85m parameters | configuration/pipeline target; not a promise that full training is practical on the current pc |

these are initial engineering configurations, not measured capability rankings. a larger randomly initialized model is not automatically smarter. data quality, token budget, training stability, evaluation, and later instruction training matter enormously.

## requirements

- python 3.10+
- pytorch 2.4+ (a recent CUDA build is recommended for the nvidia gpu)
- windows 11, linux, or macos
- licensed text files you have permission to use

## quick start (windows powershell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# install a CUDA-enabled pytorch build that matches your system from:
# https://pytorch.org/get-started/locally/
pip install -r requirements.txt

# put your own permitted .txt training files under data/train/
# then run the small model:
python -m blank_model.train --config blank --data data/train --steps 200 --output checkpoints/blank
```

## get a starter dataset

a streaming preparation script can fetch a small sample from selected public Hugging Face datasets, filter empty/short rows, remove exact duplicates, and write train/validation text files plus a manifest.

```powershell
pip install -r requirements-data.txt
python scripts/prepare_datasets.py --dataset tinystories --max-samples 5000
# coding-focused samples (review dataset cards and attribution requirements first)
python scripts/prepare_datasets.py --dataset codefeedback --max-samples 5000
python scripts/prepare_datasets.py --dataset opencodegenetic --max-samples 5000
python scripts/prepare_datasets.py --dataset codealpaca --max-samples 4000
python -m blank_model.train --config blank --data data/train/tinystories.txt --validation-data data/validation --steps 200 --output checkpoints/tinystories
```

available sources and data-handling notes are documented in [the dataset starter pack](docs/DATASET_STARTER_PACK.md). review each dataset's current card, source terms, and license before use. streaming avoids downloading the full dataset, but the loader still needs internet access and may cache data. downloaded datasets are intentionally excluded from Git.

## one-command from-scratch training (windows)

From the repository root in PowerShell, after creating the virtual environment and installing the CUDA-enabled PyTorch build as described above:

```powershell
pip install -r requirements-data.txt
powershell -ExecutionPolicy Bypass -File .\scripts\train_from_scratch.ps1
```

This prepares modest streamed samples from TinyStories, FineWeb-Edu, CodeFeedback, and CodeAlpaca, then trains the randomly initialized `blank` model. By default, data, manifests, checkpoints, and run metadata go under `$env:USERPROFILE\Downloads\blank-training-run\`, so large artifacts stay out of Git. The default run is 2,000 optimizer steps; you can adjust it, for example:

```powershell
.\scripts\train_from_scratch.ps1 -Model blank -Steps 5000
```

The script needs internet access for Hugging Face and can take a while. Review dataset licenses and source terms before use or redistribution. Training from scratch on a small sample is a real training run, but it will not produce a strong coding assistant by itself. The first goal is a stable loss curve and reproducible checkpoints.

## continue training: blank-child -> blank-preteen (windows)

After completing the initial run with the default script, you can continue the existing `blank` checkpoint on coding-focused instruction data:

```powershell
powershell -ExecutionPolicy Bypass -File .\\scripts\\train_preteen.ps1
```

The script expects the child checkpoint at `Downloads\\blank-training-run\\checkpoints\\blank\\latest.pt`. It downloads modest samples from CodeFeedback, OpenCodeGeneticInstruct, and CodeAlpaca, then resumes the same model and optimizer for 2,000 additional steps. It writes the new checkpoint to `Downloads\\blank-training-run\\checkpoints\\blank-preteen\\latest.pt`; it does not overwrite the original child checkpoint. Adjust `-Steps` and the dataset sample parameters if needed.

This is **coding-focused continued next-token training**, not yet dedicated supervised fine-tuning. It keeps the same architecture and 256-token context, so this stage tests the effect of better-targeted data rather than changing several variables at once. Review dataset cards and licenses before use.

## stage 4: blank-hybrid (coding + arithmetic + stories)

Once the pre-teen checkpoint exists, run this stage from the repository root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_hybrid.ps1
```

The hybrid script prepares up to 5,000 examples: 2,000 coding examples split across CodeFeedback and CodeAlpaca, 1,500 TinyStories continuations, and 1,500 locally generated arithmetic question/answer pairs. The actual accepted counts are written to `Downloads\blank-training-run\hybrid-data\manifest.json`; dataset streaming can return fewer usable records than requested.

It then performs 2,000 additional optimizer steps of **response-masked supervised fine-tuning**: blank sees the prompt but the loss is calculated on the answer tokens, unlike the previous stage's ordinary next-token corpus training. The original child and pre-teen checkpoints remain unchanged. The new checkpoint is `Downloads\blank-training-run\checkpoints\blank-hybrid\latest.pt`.

The model still uses byte-level tokens and a 256-token context. This hybrid objective is a more appropriate experiment for practicing answers, but it does not guarantee reliable coding or arithmetic; compare generated samples and validation loss before judging the result. Review each public dataset's current license and source terms before use or redistribution.

## smoke test

if you only want to test that the model code runs, use the included smoke test. it generates random weights and checks a forward pass; it does **not** prove language learning.

```powershell
python -m blank_model.smoke_test
```

## generate text

after training, use the checkpoint directory printed by the trainer:

```powershell
python -m blank_model.generate --checkpoint checkpoints/tinystories/latest.pt --prompt "hello"
```

the tokenizer is byte-level to keep the first version dependency-light and transparent. it is deliberately basic and inefficient compared with a trained subword tokenizer. model outputs may be incoherent until enough suitable data and training compute are available.

## data rules

- only use data you own, have permission to use, or whose license permits the intended use.
- record each source, license, download date, and any filtering in `data/manifest.csv` and generated manifests.
- do not assume that publicly accessible means freely reusable.
- keep private, personal, confidential, and credential data out of training sets.
- do not commit datasets, checkpoints, or large generated files to git.

## roadmap

1. [x] byte tokenizer and decoder-only transformer
2. [x] configs for the three model sizes
3. [x] training loop, checkpoints, and text generation
4. [x] basic smoke tests
5. [x] starter dataset preparation script and documentation
6. [ ] establish validation loss and repeatable benchmarks
7. [ ] add dedicated instruction fine-tuning and chat formatting after base training is stable
8. [ ] evaluate each model before making capability claims

## current limitations

- byte-level tokens mean long contexts cost more compute than subword tokens.
- the starter data loader reads the selected text files into memory; keep the first dataset small.
- the dataset script currently deduplicates exact matches within one run, not across every source or previously downloaded corpus.
- no pretrained weights are used. every parameter begins randomly initialized.
- this is an educational base-model pipeline, not a production chatbot or coding model.
