import torch
from torch import nn

from src.model.attention.causal_attention import CausalAttention


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The output shape is (batch_size, num_tokens, num_heads * d_out)

        # However, CausalAttention is applied num_heads times and can be more inefficient than
        #   somehow having all the computation don in unified matrix operations, instead of
        #   num_heads times.
        return torch.cat([head(x) for head in self.heads], dim=-1)
