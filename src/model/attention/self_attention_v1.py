import torch
import torch.nn.functional as F
from torch import nn


class SelfAttentionV1(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super(SelfAttentionV1, self).__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keys = torch.matmul(x, self.W_key)
        queries = torch.matmul(x, self.W_query)
        values = torch.matmul(x, self.W_value)

        # Attention scores are computed as the dot product each query vector with every key vector.
        attn_scores = torch.matmul(queries, keys.transpose(-1, -2))

        # Attention weights are scaled with the square root of the number of keys.
        attn_weights = F.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # Context vector is the weighted average of each value vector by attention weights.
        context_vec = torch.matmul(attn_weights, values)
        return context_vec
