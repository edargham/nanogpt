"""Entry point for the NanoGPT package."""

from .preprocessing import create_dataset, create_vocab_set, decode, read_text, split_data
from .models import BiGramLM
from .trainer import ModelTrainer

import torch
from torch import nn


def main():
    """Load the Shakespeare corpus, train a BiGramLM, and generate sample text.

    Reads ``data/shakespeare.txt``, encodes it into token indices, splits into
    train/validation sets, trains a ``BiGramLM`` with ``ModelTrainer`` for
    10,000 epochs, then decodes and prints 1,000 generated characters.
    """
    context_length = 8

    corpus = read_text('data/shakespeare.txt')
    print('Data loaded successfully.')

    vocab = create_vocab_set(corpus)
    print(''.join(vocab))

    data, _ = create_dataset(corpus)
    print(data, f'dtype: {data.dtype}', f'shape: {data.shape}')

    train, val = split_data(data, 0.9)

    device_str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
    device = torch.device(device_str)

    print(f'Using device: {device_str}')

    model = BiGramLM(
        len(vocab),
    )

    trainer = ModelTrainer(
        model,
        nn.CrossEntropyLoss(),
        torch.optim.AdamW(model.parameters(), lr=1e-2),
        batch_size=32,
        context_length=context_length,
        device=device
    )

    trainer.train(
        train,
        epochs=3000,
        val_data=val,
    )

    print(decode(
        model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 500)[0].tolist(),
        vocab
    ))

if __name__ == '__main__':
    main()