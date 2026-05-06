# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the project

Run the entry point from the repo root:

```bash
python -m src.nanogpt
```

The package expects data files to be referenced relative to the working directory (e.g., `data/shakespeare.txt`), so always run from the repo root.

## Architecture

This is an in-progress character-level GPT implementation. The source lives under `src/nanogpt/` as an importable package.

**`src/nanogpt/preprocessing/`** — data pipeline (the only implemented layer so far):
- `loader.py`: reads raw text files from disk
- `language.py`: character-level tokenization — `create_vocab_set` builds a sorted character vocabulary, `encode` maps chars → int indices, `decode` maps indices → chars using a simple lookup table. (`create_vocab_set` has an unused `char_lvl` parameter.)
- `data_prep.py`: dataset utilities —
  - `create_dataset(corpus)` — encodes the full corpus into a `torch.long` tensor and returns `(data, vocab)`
  - `split_data(data, train_size=0.8)` — splits the encoded tensor into train/validation portions
  - `make_batches(data, batch_size=4, context_length=8)` — samples random `(x, y)` batch pairs; hardcodes `torch.manual_seed(1337)` (intentional for reproducibility during development)
- `__init__.py` exports: `create_dataset`, `create_vocab_set`, `make_batches`, `read_text`, `split_data`

**`src/nanogpt/__main__.py`** — entry point; loads the Shakespeare corpus, builds the vocab, encodes the full dataset into a tensor, and samples random indices. All imports use relative form (`from .preprocessing import ...`). The model and training loop do not exist yet.

**`data/shakespeare.txt`** — the training corpus (Tiny Shakespeare dataset).
