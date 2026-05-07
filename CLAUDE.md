# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the project

Run the entry point from the repo root:

```bash
python -m src.nanogpt
```

The package expects data files to be referenced relative to the working directory (e.g., `data/shakespeare.txt`), so always run from the repo root.

## Architecture

This is an in-progress character-level GPT implementation. The source lives under `src/nanogpt/` as an importable package with three subpackages.

**`src/nanogpt/preprocessing/`** — data pipeline:
- `loader.py`: reads raw text files from disk
- `language.py`: character-level tokenization — `create_vocab_set` builds a sorted character vocabulary, `encode` maps chars → int indices, `decode` maps indices → chars. (`create_vocab_set` has an unused `char_lvl` parameter.)
- `data_prep.py`:
  - `create_dataset(corpus)` — encodes the full corpus into a `torch.long` tensor and returns `(data, vocab)`
  - `split_data(data, train_size=0.8)` — splits the encoded tensor into train/validation portions
  - `make_batches(data, batch_size=4, context_length=8)` — samples random `(x, y)` batch pairs; hardcodes `torch.manual_seed(1337)` (intentional for reproducibility during development)

**`src/nanogpt/models/`** — model definitions:
- `bigramlm.py`: `BiGramLM(nn.Module)` — a simple bigram language model backed by a single `nn.Embedding` of shape `(vocab_size, vocab_size)`. `forward(x)` returns raw logits of shape `(B, T, vocab_size)`. Sets `torch.manual_seed(1337)` at init.

**`src/nanogpt/trainer/`** — training loop:
- `model_trainer.py`: `ModelTrainer` — wraps a model, loss function, and optimizer. `_perform_step(x, y)` handles device transfer, forward pass, logit reshaping to `(B*T, C)`, and loss computation. `train(train_data, epochs, val_data=None)` iterates epochs, delegating to `_perform_step` for both the training and optional validation pass, then back-propagates and updates parameters. Progress is shown via a per-epoch `tqdm` bar.

**`src/nanogpt/__main__.py`** — entry point: loads Shakespeare corpus → tokenizes → encodes → splits → instantiates `BiGramLM` + `ModelTrainer` → trains for 100 epochs with `AdamW(lr=1e-3)` and `CrossEntropyLoss`. Device selection prefers CUDA, then MPS, then CPU.

**`data/shakespeare.txt`** — the training corpus (Tiny Shakespeare dataset).

## Known issues / design notes

- `make_batches` always resets `torch.manual_seed(1337)` before sampling, so every call returns the same batch. This is intentional during early development but must be removed for real training.
- `ModelTrainer.train` calls `make_batches` without passing `context_length` on the training step (only on the validation step), so training always uses the default `context_length=8`. This is in `_perform_step`'s caller, not in `_perform_step` itself.