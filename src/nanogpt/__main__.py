"""Entry point for the NanoGPT package."""

from .preprocessing import create_dataset, create_vocab_set, decode, read_text, split_data
from .models import NanoGPT
from .trainer import ModelTrainer

import torch
from torch import nn
torch.manual_seed(1337)

def main():
    """Load the Shakespeare corpus, train a BiGramLM, and generate sample text.

    Reads ``data/shakespeare.txt``, encodes it into token indices, splits into
    train/validation sets, trains a ``BiGramLM`` with ``ModelTrainer`` for
    5,000 epochs, then decodes and prints 1,000 generated characters.
    """
    context_length = 8
    num_embeddings = 32
    num_attn_heads = 4

    batch_size = 32
    lr = 1e-2
    epochs = 5000

    num_tokens_to_generate = 500

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

    model = NanoGPT(
        len(vocab),
        num_embed=num_embeddings,
        context_length=context_length,
        num_heads=num_attn_heads,
        device=device,
    )

    trainer = ModelTrainer(
        model,
        nn.CrossEntropyLoss(),
        torch.optim.AdamW(model.parameters(), lr=lr),
        batch_size=batch_size,
        context_length=context_length,
        device=device
    )

    trainer.train(
        train,
        epochs=epochs,
        val_data=val,
    )

    print(decode(
        model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), num_tokens_to_generate)[0].tolist(),
        vocab
    ))

if __name__ == '__main__':
    main()