from torch import nn

from src.model.attention.multi_head_attention import MultiHeadAttention
from src.model.layer.feed_forward_equal import FeedForwardEqual
from src.model.layer.layer_normalization import LayerNormalization


class TransformerBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        context_length: int,
        num_heads: int,
        dropout: float = 0,
        qkv_bias: bool = False,
        hidden_size=None,
    ):
        super(TransformerBlock, self).__init__()

        hidden_size = hidden_size if hidden_size else 4 * emb_dim

        self.attn = MultiHeadAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_length=context_length,
            num_heads=num_heads,
            dropout=dropout,
            qkv_bias=qkv_bias,
        )
        self.ff = FeedForwardEqual(emb_dim, hidden_size)
        self.norm1 = LayerNormalization(emb_dim)
        self.norm2 = LayerNormalization(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut

        return x
