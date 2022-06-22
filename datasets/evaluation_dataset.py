from torch.utils.data import DataLoader, Dataset

import torch


__all__ = ["EvaluationDataset"]

class EvaluationDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]
