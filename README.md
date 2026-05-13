# NanoGPT

A character-level GPT implementation in PyTorch, built from scratch as a learning exercise following Andrej Karpathy's lecture [**Let's build GPT: from scratch, in code, spelled out**](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3726s&pp=ygUUbGV0J3MgY3JlYXRlIGNoYXRncHQ%3D). The companion repository for the original lecture is [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture).

The model trains on the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset and generates Shakespeare-style text character by character.

---

## Architecture

The implementation follows the decoder-only transformer architecture described in [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), adapted for character-level language modelling.

```
Input tokens (B, T)
       │
       ├─ Token Embedding        (vocab_size → num_embed)
       ├─ Positional Embedding   (context_length → num_embed)
       │              ↓
       │        [sum] (B, T, num_embed)
       │              │
       │    ┌─────────┴──────────┐
       │    │   TransformerBlock │  × num_layers
       │    │  ┌──────────────── │
       │    │  │  LayerNorm      │
       │    │  │  MultiHeadAttn  │  (causal mask)
       │    │  │  + residual     │
       │    │  │  LayerNorm      │
       │    │  │  FeedForward    │
       │    │  │  + residual     │
       │    └──┴─────────────────┘
       │              │
       │         LayerNorm
       │              │
       │       Linear head       (num_embed → vocab_size)
       │              │
       └──────► Logits (B, T, vocab_size)
```

### Components

| Module | File | Description |
|--------|------|-------------|
| `NanoGPT` | `models/gpt.py` | Top-level decoder-only transformer |
| `TransformerBlock` | `models/layers/transformer_block.py` | Pre-norm residual block; supports decoder-only and full encoder-decoder modes |
| `MultiHeadAttention` | `models/layers/multi_head_attention.py` | Parallel attention heads with output projection |
| `SelfAttention` | `models/layers/self_attention.py` | Single scaled dot-product attention head with optional causal mask |
| `FeedForward` | `models/layers/feed_forward.py` | Position-wise two-layer MLP (4× expansion, ReLU) |
| `BiGramLM` | `models/bigramlm.py` | Simple bigram baseline (retained for reference) |
| `BaseLM` | `models/baselm.py` | Abstract base class; provides the autoregressive `generate` loop |
| `ModelTrainer` | `trainer/model_trainer.py` | Training loop with periodic loss estimation |

---

## Project Structure

```
NanoGPT/
├── data/
│   └── shakespeare.txt          # Tiny Shakespeare training corpus
└── src/nanogpt/
    ├── __main__.py               # Entry point
    ├── models/
    │   ├── baselm.py
    │   ├── bigramlm.py
    │   ├── gpt.py
    │   └── layers/
    │       ├── self_attention.py
    │       ├── multi_head_attention.py
    │       ├── feed_forward.py
    │       └── transformer_block.py
    ├── preprocessing/
    │   ├── loader.py
    │   ├── language.py
    │   └── data_prep.py
    └── trainer/
        └── model_trainer.py
```

---

## Requirements

- Python >= 3.11
- [PyTorch](https://pytorch.org/get-started/locally/)
- `tqdm`
- `numpy`

Install dependencies:

```bash
pip install torch tqdm numpy
```

---

## Usage

Run from the repository root:

```bash
python -m src.nanogpt
```

The entry point expects `data/shakespeare.txt` to be present relative to the working directory.

### Default hyperparameters

| Parameter | Value |
|-----------|-------|
| `context_length` | 128 |
| `num_embed` | 192 |
| `num_heads` | 4 |
| `num_layers` | 4 |
| `batch_size` | 32 |
| `learning_rate` | 3e-4 |
| `epochs` | 5 000 |
| `train/val split` | 90 / 10 |
| `tokens generated` | 500 |

Device selection is automatic: CUDA → MPS → CPU.

---

## References

- Karpathy, A. — [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3726s&pp=ygUUbGV0J3MgY3JlYXRlIGNoYXRncHQ%3D) (YouTube, 2023)
- Karpathy, A. — [ng-video-lecture](https://github.com/karpathy/ng-video-lecture) (GitHub)
- Vaswani et al. — [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
