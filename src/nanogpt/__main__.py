"""Entry point for the NanoGPT package."""

from .preprocessing import create_dataset, create_vocab_set, read_text, split_data
from .models import BiGramLM
from .trainer import ModelTrainer

import torch
from torch import nn


def main():
    """Load the Shakespeare corpus, tokenize it, and print diagnostic output."""
    corpus = read_text('data/shakespeare.txt')
    print(corpus)

    vocab = create_vocab_set(corpus)
    print(''.join(vocab))

    data, _ = create_dataset(corpus)
    print(data, f'dtype: {data.dtype}', f'shape: {data.shape}')

    train, val = split_data(data, 0.8)

    device_str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
    device = torch.device(device_str)

    print(f'Using device: {device_str}')

    model = BiGramLM(len(vocab))

    trainer = ModelTrainer(
        model,
        nn.CrossEntropyLoss(),
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        batch_size=32,
        context_length=8,
        device=device
    )

    trainer.train(
        train,
        epochs=100,
        val_data=val,
    )

if __name__ == '__main__':
    main()