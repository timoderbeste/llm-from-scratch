import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
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
        assert d_out % num_heads == 0, 'd_out must be divisible by num_heads'

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Note that here three separate weight matrices are used, which results in
        #   separate computation of queries, keys, and values.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            'mask',
            torch.tril(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # (batch_size, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # (batch_size, num_tokens, num_heads, head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        # (batch_size, num_heads, num_tokens, head_dim)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        # Compute dot-product attention for each head:
        # (batch_size, num_heads, num_tokens, head_dim) x
        # (batch_size, num_heads, head_dim, num_tokens) ->
        # (batch_size, num_heads, num_tokens, num_tokens).
        attn_scores = torch.matmul(queries, keys.transpose(2, 3))
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        context_vec = _compute_and_apply_attention_weights(
            attn_scores,
            keys,
            values,
            batch_size,
            num_tokens,
            self.d_out,
            self.dropout,
            self.out_proj
        )

        return context_vec


class MultiHeadAttentionWithCombinedQKV(nn.Module):
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

        assert d_out % num_heads == 0, 'd_out must be divisible by num_heads'

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            'mask',
            torch.tril(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        qkv = _compute_qkv(self.qkv, x, self.num_heads, self.head_dim)
        queries, keys, values = qkv

        attn_scores = torch.matmul(qkv, keys.transpose(3, 4))
        attn_scores = attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        context_vec = _compute_and_apply_attention_weights(
            attn_scores,
            keys,
            values,
            batch_size,
            num_tokens,
            self.d_out,
            self.dropout,
            self.out_proj
        )

        return context_vec


class MHAPyTorchScaledDotProduct(nn.Module):
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

        assert d_out % num_heads == 0, 'd_out must be divisible by num_heads'

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        qkv = _compute_qkv(self.qkv, x, self.num_heads, self.head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout
        context_vec = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=None,
            dropout_p=use_dropout,
            is_causal=True
        )
        context_vec = (context_vec
                       .transpose(1, 2)
                       .contiguous()
                       .view(batch_size, num_tokens, self.d_out))
        context_vec = self.proj(context_vec)
        return context_vec


def _compute_qkv(qkv, x, num_heads, head_dim):
    batch_size, num_tokens, d_in = x.shape
    qkv = qkv(x)

    # (batch_size, num_tokens, 3 * d_in) -> (batch_size, num_tokens, 3, num_heads, head_dim)
    qkv = qkv.view(batch_size, num_tokens, 3, num_heads, head_dim)

    # (batch_size, num_tokens, 3, num_heads, head_dim) ->
    # (3, batch_size, num_heads, num_tokens, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    return qkv


def _compute_and_apply_attention_weights(
    attn_scores,
    keys,
    values,
    batch_size,
    num_tokens,
    d_out,
    dropout,
    out_proj
):
    attn_weights = F.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
    attn_weights = dropout(attn_weights)
    # (batch_size, num_tokens, num_heads, head_dim)
    context_vec = torch.matmul(attn_weights, values).transpose(1, 2)
    # (batch_size, num_tokens, d_out)
    # contiguous() makes sure that all elements of a tensor is placed in a contiguous segment
    #   of the memory.
    context_vec = context_vec.contiguous().view(batch_size, num_tokens, d_out)
    context_vec = out_proj(context_vec)
    return context_vec
