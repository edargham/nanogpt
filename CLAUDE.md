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
  - `make_batches(data, batch_size, context_length)` — samples random `(x, y)` batch pairs; hardcodes `torch.manual_seed(1337)` (intentional for reproducibility during development)

**`src/nanogpt/models/`** — model definitions:
- `baselm.py`: `BaseLM(nn.Module, ABC)` — abstract base class all models must implement. Declares two abstract methods: `forward(x)` and `generate(x, max_new_tokens)`.
- `bigramlm.py`: `BiGramLM(BaseLM)` — a bigram language model backed by a single `nn.Embedding` of shape `(vocab_size, vocab_size)`. `forward(x)` returns raw logits `(B, T, vocab_size)`. `generate(x, max_new_tokens)` autoregressively samples tokens by taking the last timestep's logits, applying softmax, and sampling via `torch.multinomial`. Sets `torch.manual_seed(1337)` at init.

**`src/nanogpt/trainer/`** — training loop:
- `model_trainer.py`: `ModelTrainer` — wraps a `BaseLM`, loss function, optimizer, `batch_size`, and `context_length`. `_perform_step(x, y)` handles device transfer, forward pass, logit reshaping to `(B*T, C)`, and loss computation. `train(train_data, epochs, val_data=None)` iterates epochs, calling `make_batches` each step using the stored `batch_size`/`context_length`, back-propagates, and updates parameters. Progress is shown via a per-epoch `tqdm` bar.

**`src/nanogpt/__main__.py`** — entry point: loads Shakespeare corpus → tokenizes → encodes → splits → instantiates `BiGramLM` + `ModelTrainer` (batch_size=32, context_length=8) → trains for 10,000 epochs with `AdamW(lr=1e-3)` and `CrossEntropyLoss` → calls `model.generate()` to produce 1,000 characters and prints decoded output. Device selection prefers CUDA, then MPS, then CPU.

**`data/shakespeare.txt`** — the training corpus (Tiny Shakespeare dataset).

## Known issues / design notes

- `make_batches` always resets `torch.manual_seed(1337)` before sampling, so every call returns the same batch. This is intentional during early development but must be removed for real training.
- New models must subclass `BaseLM` and implement both `forward` and `generate`.