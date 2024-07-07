import tiktoken
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from src.utils.text_tokenization import text_to_token_ids, token_ids_to_text


def generate_text_simple(
    model: nn.Module,
    token_idxes: Tensor,
    max_new_tokens: int,
    context_size: int
) -> Tensor:
    for _ in range(max_new_tokens):
        idx_cond = token_idxes[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)

        idx_next = torch.argmax(probs, dim=-1, keepdim=True)

        token_idxes = torch.cat((token_idxes, idx_next), dim=1)

    return token_idxes


def generate_and_print_sample(
    model: torch.nn.Module,
    tokenizer: tiktoken.Encoding,
    device: str,
    start_context: str,
    context_length: int,
) -> None:
    model.eval()
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model, encoded, 50, context_length)
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace('\n', ' '))
