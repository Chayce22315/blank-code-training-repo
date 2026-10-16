# dataset workflow

1. put permitted UTF-8 text files in `data/train/`.
2. put a separate held-out set in `data/validation/`; do not copy the same documents into both.
3. add one row per source file or source collection to `data/manifest.csv`, including the license and source URL/name.
4. remove secrets, private records, and data you do not have permission to use.
5. deduplicate documents where practical and note the filtering steps.
6. keep the actual corpus out of git. `.gitignore` excludes files under the training and validation folders.

## quick pipeline check

The repository includes `examples/toy_corpus.txt`, a tiny synthetic document intended only to test that the training loop can save and reload a checkpoint:

```powershell
python -m blank_model.train --config blank --data examples/toy_corpus.txt --steps 20 --output checkpoints/toy
python -m blank_model.generate --checkpoint checkpoints/toy/latest.pt --prompt "blank is" --tokens 80
```

This toy run is not meaningful pretraining and will not create a useful assistant. Use it to validate the plumbing, then replace it with a larger, licensed corpus.
