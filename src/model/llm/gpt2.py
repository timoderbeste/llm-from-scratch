import torch
from torch import nn

from src.model.block.transformer_block import TransformerBlock
from src.model.layer.layer_normalization import LayerNormalization


class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.tok_emb = nn.Embedding(config['vocab_size'], config['emb_dim'])
        self.pos_emb = nn.Embedding(config['context_length'], config['emb_dim'])
        self.drop_emb = nn.Dropout(config['dropout'])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(
                config['emb_dim'],
                config['emb_dim'],
                config['num_heads'],
                config['dropout']
            ) for _ in
              range(config['num_layers'])],
        )
        self.final_norm = LayerNormalization(config['emb_dim'])
        self.out_head = nn.Linear(config['emb_dim'], config['vocab_size'], bias=False)

    def forward(self, in_idxes):
        # FIXME: Somehow the computed values are all NaN!!!
        batch_size, seq_len = in_idxes.size()
        tok_embeds = self.tok_emb(in_idxes)
        pos_embeds = self.pos_emb(torch.arange(seq_len).to(in_idxes.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
