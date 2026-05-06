"""Dataset construction and batching utilities."""

from .language import encode, create_vocab_set
from typing import Tuple, List
import torch

def create_dataset(corpus: str) -> Tuple[torch.Tensor, List[str]]:
    """Encode a raw text corpus into a tensor of token indices.

    Args:
        corpus: Full raw text to tokenize.

    Returns:
        A tuple ``(data, vocab)`` where ``data`` is a ``torch.long`` tensor of
        token indices and ``vocab`` is the sorted character vocabulary derived
        from the corpus.
    """
    vocab = create_vocab_set(corpus)
    data = torch.tensor(encode(corpus, vocab), dtype=torch.long)
    return data, vocab

def split_data(data: torch.Tensor, train_size: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split an encoded dataset into training and validation portions.

    Args:
        data: 1-D tensor of token indices for the full dataset.
        train_size: Fraction of tokens assigned to the training split.
            Must be in (0, 1). Defaults to 0.8.

    Returns:
        A tuple ``(train, val)`` of contiguous tensor slices.
    """
    n = train_size*len(data)
    return data[:n], data[n:]

def make_batches(data: torch.Tensor, batch_size: int=4, context_length: int=8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of input/target sequence pairs.

    Randomly draws ``batch_size`` starting positions and extracts overlapping
    context windows so that ``y`` is ``x`` shifted one step to the right,
    forming the next-token prediction targets.

    Args:
        data: 1-D tensor of token indices to sample from.
        batch_size: Number of independent sequences per batch. Defaults to 4.
        context_length: Number of tokens per sequence. Defaults to 8.

    Returns:
        A tuple ``(x, y)`` of shape ``(batch_size, context_length)``, where
        ``y[i, t]`` is the token that follows ``x[i, t]`` in the corpus.
    """
    torch.manual_seed(1337)
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])
    return x, y