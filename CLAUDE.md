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
- `baselm.py`: `BaseLM(nn.Module, ABC)` — abstract base class. `forward(x)` is abstract. `generate(x, max_new_tokens, scope_context=False)` is a concrete implementation that autoregressively samples tokens: when `scope_context=True` crops the input to the last `context_length` tokens before each forward pass, takes the last timestep's logits, applies softmax, and samples via `torch.multinomial`.
- `bigramlm.py`: `BiGramLM(BaseLM)` — retained but not used by the current entry point.
- `gpt.py`: `NanoGPT(BaseLM)` — the active model. Takes `vocab_size`, `num_embed`, `context_length`, `num_heads`, `num_layers`, and `device`. Stacks token + positional embeddings, `num_layers` `TransformerBlock`s, a final `nn.LayerNorm`, and a linear projection head (`num_embed → vocab_size`). `forward(x)` returns logits `(B, T, vocab_size)`. `generate` delegates to `BaseLM.generate` with `scope_context=True`.

**`src/nanogpt/models/layers/`** — reusable transformer components:
- `self_attention.py`: `SelfAttention` — single scaled dot-product attention head. Params: `head_size`, `num_embed`, `context_length`, `mask=True`, `dropout=0.2`. When `mask=True`, registers a causal `tril` buffer and applies it before softmax. Scaling factor is `head_size**-0.5`.
- `multi_head_attention.py`: `MultiHeadAttention` — runs `num_heads` `SelfAttention` heads in parallel (each with `head_size = num_embed // num_heads`), concatenates outputs along the feature dim, projects back to `num_embed` via a linear layer, then applies dropout.
- `feed_forward.py`: `FeedForward` — two-layer MLP: `num_embed → 4*num_embed` (ReLU) `→ num_embed` (Dropout). Applied position-wise after attention.
- `transformer_block.py`: `TransformerBlock` — pre-norm residual block. Params: `num_heads`, `num_embed`, `context_length`, `decoder_only=True`. Always has a masked `MultiHeadAttention` + `FeedForward` with residual connections and `LayerNorm` before each sub-layer. When `decoder_only=False`, adds a second unmasked `MultiHeadAttention` (encoder-style cross-attention path). Allocates exactly 2 `LayerNorm`s in decoder-only mode, 3 otherwise, stored in `nn.ModuleList`.

**`src/nanogpt/trainer/`** — training loop:
- `model_trainer.py`: `ModelTrainer` — wraps a `BaseLM`, loss function, optimizer, `batch_size`, `context_length`, and `device`. `_perform_step(x, y)` handles device transfer, forward pass, logit reshaping to `(B*T, C)`, and loss computation. `_estimate_loss(data, eval_iters=200)` averages the loss over `eval_iters` random batches under `torch.no_grad()`. `train(train_data, epochs, val_data=None, eval_interval=300, eval_iters=200)` iterates epochs with a `tqdm` bar; every `eval_interval` steps it calls `_estimate_loss` on train (and optionally val) data and updates the bar's postfix with `loss` and `val_loss`.

**`src/nanogpt/__main__.py`** — entry point: loads Shakespeare corpus → builds vocab → encodes → splits 90/10 → instantiates `NanoGPT` (context_length=128, num_embed=192, num_heads=4, num_layers=4) + `ModelTrainer` (batch_size=32, lr=3e-4) → trains for 5,000 epochs with `AdamW` and `CrossEntropyLoss` → generates 500 tokens from a zero seed and prints the decoded output. Device selection prefers CUDA, then MPS, then CPU. `torch.manual_seed(1337)` is set globally at module load.

**`data/shakespeare.txt`** — the training corpus (Tiny Shakespeare dataset).

## Known issues / design notes

- `make_batches` always resets `torch.manual_seed(1337)` before sampling, so every call returns the same batch. This is intentional during early development but must be removed for real training.
- New models must subclass `BaseLM` and implement `forward`. `generate` can be inherited from `BaseLM` or overridden.