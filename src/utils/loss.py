from typing import Tuple

import torch

import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader


def compute_batch_loss(
    input_batch: Tensor,
    target_batch: Tensor,
    model: nn.Module,
    device: str
) -> Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def compute_full_loss(
    data_loader: DataLoader,
    model: nn.Module,
    device: str,
    num_batches: 0 = None
) -> Tensor:
    total_loss = torch.tensor(0.0, device=device)
    if len(data_loader) == 0:
        return torch.tensor(torch.nan)
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    input_batch: Tensor
    target_batch: Tensor
    for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
        if batch_idx < num_batches:
            loss = compute_batch_loss(input_batch, target_batch, model, device)
            total_loss += loss
        else:
            break
    return total_loss / num_batches


def evaluate_model_loss(
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    model: nn.Module,
    device: str,
    eval_iter: int
) -> Tuple[Tensor, Tensor]:
    model.eval()
    with torch.no_grad():
        train_loss = compute_full_loss(train_data_loader, model, device, eval_iter)
        val_loss = compute_full_loss(val_data_loader, model, device, eval_iter)
    return train_loss, val_loss
