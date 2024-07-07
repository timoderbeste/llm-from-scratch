from typing import List

import matplotlib.pyplot as plt
from torch import Tensor


def plot_losses(
    epochs: Tensor,
    tokens_seen: List = None,
    train_losses: List[Tensor] = None,
    val_losses: List[Tensor] = None
):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    if train_losses is not None:
        ax1.plot(
            epochs, train_losses, label='Train Loss', color='tab:blue'
        )
    if val_losses is not None:
        ax1.plot(
            epochs, val_losses, label='Val Loss', color='tab:red', linestyle='-.'
        )
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    
    if tokens_seen is not None:
        ax2 = ax1.twiny()
        ax2.plot(tokens_seen, train_losses, label='Train Loss')
        ax2.set_xlabel('Tokens seen')

    fig.tight_layout()
    plt.show()
