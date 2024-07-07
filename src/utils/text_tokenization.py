import torch
from tiktoken import Encoding
from torch import Tensor

from src.dataset import END_OF_TEXT_TOKEN


def text_to_token_ids(text: str, tokenizer: Encoding):
    encoded = tokenizer.encode(text, allowed_special={END_OF_TEXT_TOKEN})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids: Tensor, tokenizer: Encoding):
    flat = token_ids.squeeze(0).tolist()
    return tokenizer.decode(flat)