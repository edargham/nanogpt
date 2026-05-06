"""Entry point for the NanoGPT package."""

from .preprocessing import create_dataset, create_vocab_set, read_text, split_data
import torch

def main():
    """Load the Shakespeare corpus, tokenize it, and print diagnostic output."""
    corpus = read_text('data/shakespeare.txt')
    print(corpus)

    vocab = create_vocab_set(corpus)
    print(''.join(vocab))

    data, _ = create_dataset(corpus)
    print(data, f'dtype: {data.dtype}', f'shape: {data.shape}')

    ix = torch.randint(len(data) - 8, (4,))
    print(ix)

if __name__ == '__main__':
    main()