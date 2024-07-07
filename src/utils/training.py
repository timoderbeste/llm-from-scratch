from typing import Tuple, List

import tiktoken
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.utils.loss import compute_batch_loss, evaluate_model_loss
from src.utils.text_generation import generate_and_print_sample


def train_model_simple(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer: tiktoken.Encoding,
    context_length: int
) -> Tuple[List[Tensor], List[Tensor], List[int]]:
    input_batch: Tensor
    target_batch: Tensor

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        for input_batch, target_batch in train_loader:
            model.train()
            optimizer.zero_grad()
            loss = compute_batch_loss(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model_loss(
                    train_loader,
                    val_loader,
                    model,
                    device,
                    eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f'Epoch {epoch + 1} (Step {global_step:06d}:')
                print(f'Train Loss: {train_loss:.4f}')
                print(f'Val Loss: {val_loss:.4f}')
        generate_and_print_sample(model, tokenizer, device, start_context, context_length)

    return train_losses, val_losses, track_tokens_seen
