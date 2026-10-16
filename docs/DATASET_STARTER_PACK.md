# blank dataset starter pack

This guide describes a small first dataset mix for experimenting with the blank model family. It is not a claim that these datasets alone are enough to create a capable assistant.

## Recommended order

1. **TinyStories** for a tiny pipeline check and simple narrative text. Dataset card: https://huggingface.co/datasets/roneneldan/TinyStories
2. **FineWeb-Edu** for educational web text. Start with a small streamed sample; the full dataset is enormous. Dataset card: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
3. **Smol-SmolTalk** for assistant-style conversations. This is supervised fine-tuning style data, not a replacement for broad pretraining text. Dataset card: https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk

Check the linked dataset card, source terms, attribution requirements, and current revision before using any dataset. Licenses and terms can vary by source or change over time. Do not redistribute raw datasets or a derived bundle unless their terms allow it. Keep source URLs, revisions, dates, filtering settings, and license notes with the dataset.

## Install

From the repository root on Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-data.txt
```

Install a CUDA-enabled PyTorch build separately using the official PyTorch selector if needed: https://pytorch.org/get-started/locally/

## Download a small sample

```powershell
# Small language-modeling sample for validating the data pipeline
python scripts/prepare_datasets.py --dataset tinystories --max-samples 5000

# Small educational-text sample
python scripts/prepare_datasets.py --dataset fineweb-edu --max-samples 10000

# Small assistant-conversation sample
python scripts/prepare_datasets.py --dataset smol-smoltalk --max-samples 5000
```

The script streams rows, filters empty/short entries, removes exact duplicates, makes a deterministic train/validation split, and writes UTF-8 text files under `data/train/` and `data/validation/`. It also writes a JSON manifest. It does not upload your data.

The first run needs internet access and may need to download dataset metadata/cache. Streaming avoids downloading the entire dataset, but does not guarantee zero network or cache usage. Choose a modest sample count first.

## Train

The current trainer is a **plain next-token language-modeling trainer**. It does not yet implement a dedicated instruction fine-tuning loss or chat-template handling. You can experiment with the text export like this:

```powershell
python -m blank_model.train --config blank --data data/train/tinystories.txt --steps 200 --output checkpoints/tinystories
```

For validation, use the trainer's validation-data option if supported by your checked-out version. Do not assume a run succeeded until its logs and checkpoint are inspected. Tiny training runs validate the pipeline, not assistant quality.

## Before building a serious corpus

- Separate pretraining text, instruction conversations, code, and evaluation data.
- Keep validation/test material out of training.
- Deduplicate across sources, not only within one download.
- Filter malformed text, spam, personal data, and unsafe or low-quality material.
- Preserve source-level provenance and applicable license notices.
- Never commit downloaded datasets, caches, tokens, or checkpoints to Git.
- Review the trainer's text-file interface before combining multiple datasets; concatenation weights sources by bytes/tokens, not by dataset count.
