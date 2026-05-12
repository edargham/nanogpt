"""Training loop for NanoGPT models."""

import torch
from torch import nn

import math
from tqdm import tqdm

from ..preprocessing import make_batches
from ..models import BaseLM


class ModelTrainer:
    """Manages the training and validation loop for a language model.

    Args:
        model: The language model to train.
        loss: Loss function applied to flattened logits and targets.
        optimizer: Optimizer used to update model parameters.
        batch_size: Number of independent sequences sampled per step.
        context_length: Number of tokens per sequence.
        device: Device on which to run the model. Defaults to CPU.
    """

    def __init__(
        self,
        model: BaseLM,
        loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        context_length: int,
        device: torch.device=torch.device("cpu"),
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.context_length = context_length
        self.model = model.to(device)
        self.device = device

    def _perform_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Run a single forward pass and compute the loss.

        Moves ``x`` and ``y`` to ``self.device``, runs the model, flattens the
        logits and targets to ``(B*T, C)`` and ``(B*T,)`` respectively, then
        applies the loss function.

        Args:
            x: Integer token indices of shape ``(B, T)``.
            y: Target token indices of shape ``(B, T)``.

        Returns:
            Scalar loss tensor for this step.
        """
        x, y = x.to(self.device), y.to(self.device)
        yh = self.model.forward(x)

        b, t, c = yh.shape
        yb, yt = y.shape

        yh = yh.view(b * t, c)
        y = y.view(yb * yt)

        loss = self.loss(yh, y)

        return loss

    @torch.no_grad()
    def _estimate_loss(self, data: torch.Tensor, eval_iters: int = 200) -> float:
        """Average loss over eval_iters random batches."""
        self.model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = make_batches(data, self.batch_size, self.context_length)
            losses[k] = self._perform_step(x, y).item()
        self.model.train()
        return losses.mean().item()

    def train(self, train_data: torch.Tensor, epochs: int, val_data: torch.Tensor = None):
        """Run the training loop for a fixed number of epochs.

        Each epoch samples one batch from ``train_data`` and delegates the
        forward pass and loss computation to ``_perform_step``, then
        back-propagates and updates model parameters. If ``val_data`` is
        provided, a validation loss is computed after each training step.
        Progress is displayed via a per-epoch ``tqdm`` bar.

        Args:
            train_data: 1-D tensor of encoded training token indices.
            epochs: Number of training epochs to run.
            val_data: Optional 1-D tensor of encoded validation token indices.
        """
        val_loss = float('nan')
        train_loss_est = float('nan')
        eval_interval = 300
        eval_iters = 200

        with tqdm(total=epochs, desc="Training", unit="step") as pbar:
            self.model.train()
            for epoch in range(epochs):
                # Training step
                self.optimizer.zero_grad(set_to_none=True)
                current_x, current_y = make_batches(train_data, self.batch_size, self.context_length)
                loss = self._perform_step(current_x, current_y)
                loss.backward()
                self.optimizer.step()

                # Averaged loss estimation at fixed interval
                if (epoch + 1) % eval_interval == 0:
                    train_loss_est = self._estimate_loss(train_data, eval_iters)
                    if val_data is not None:
                        val_loss = self._estimate_loss(val_data, eval_iters)

                    # Update bar every step
                    postfix = {
                        "loss": f"{train_loss_est:.4f}" if not isinstance(train_loss_est, float) or not __import__(
                            'math').isnan(train_loss_est) else "...",
                        "val_loss": f"{val_loss:.4f}" if val_data is not None and not math.isnan(
                            val_loss) else ("..." if val_data is not None else None),
                    }

                    # Drop val_loss key entirely if no val_data
                    if val_data is None:
                        postfix.pop("val_loss")

                    pbar.set_postfix(postfix)
                pbar.update(1)