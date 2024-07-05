import torch
import torch.nn.functional as F
from torch import nn


class SelfAttentionV2(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super(SelfAttentionV2, self).__init__()

        # Using nn.Linear over manual usage of nn.Parameter and matmul leads to more stable
        #   model training due to preferred weight initialization: kaiming_uniform_.
        self.W_query = nn.Linear(d_in, d_out)
        self.W_key = nn.Linear(d_in, d_out)
        self.W_value = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Attention scores are computed as the dot product each query vector with every key vector.
        attn_scores = torch.matmul(queries, keys.transpose(-1, -2))

        # Attention weights are scaled with the square root of the number of keys.
        attn_weights = F.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # Context vector is the weighted average of each value vector by attention weights.
        context_vec = torch.matmul(attn_weights, values)
        return context_vec
