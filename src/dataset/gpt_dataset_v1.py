from typing import Tuple

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

from src.dataset import END_OF_TEXT_TOKEN


class GPTDatasetV1(Dataset):
    def __init__(
        self,
        text: str,
        tokenizer: tiktoken.Encoding,
        max_len: int = 256,
        stride: int = 128,
    ):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text, allowed_special={END_OF_TEXT_TOKEN})

        for idx in range(0, len(token_ids) - max_len, stride):
            input_chunk = token_ids[idx:idx + max_len]
            target_chunk = token_ids[idx + 1:idx + max_len + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        assert len(self.input_ids) == len(self.target_ids)
        return len(self.input_ids)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(self.input_ids) == len(self.target_ids)
        return self.input_ids[idx], self.target_ids[idx]

    def to_data_loader(
        self,
        batch_size=4,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )
