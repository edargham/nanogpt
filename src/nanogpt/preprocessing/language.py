"""Character-level tokenization utilities."""

from typing import List

def create_vocab_set(corpus: str, char_lvl: bool = False) -> List[str]:
    """Build a sorted vocabulary list from a text corpus.

    Args:
        corpus: Raw text from which to extract unique characters.
        char_lvl: Unused; reserved for future sub-character tokenization modes.

    Returns:
        Sorted list of unique characters appearing in the corpus.
    """
    return sorted(list(set(corpus)))

def encode(corpus: str, vocab: List[str]) -> List[int]:
    """Map each character in a string to its integer index in the vocabulary.

    Args:
        corpus: Text to encode.
        vocab: Ordered vocabulary list produced by ``create_vocab_set``.

    Returns:
        List of integer token indices, one per character in ``corpus``.
    """
    lookup_t = {ch: i for i, ch in enumerate(vocab)}
    return [lookup_t[ch] for ch in corpus]

def decode(tokens: List[int], vocab: List[str]) -> str:
    """Convert a sequence of token indices back into a string.

    Args:
        tokens: List of integer indices to decode.
        vocab: Ordered vocabulary list produced by ``create_vocab_set``.

    Returns:
        Reconstructed string corresponding to the token sequence.
    """
    lookup_t = {i: ch for i, ch in enumerate(vocab)}
    return ''.join(lookup_t[tok] for tok in tokens)