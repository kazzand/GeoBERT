from torch.utils.data import DataLoader, Dataset

import torch

__all__ = ["CustomMetricLearningDataset"]


class CustomMetricLearningDataset(Dataset):
    def __init__(self, df, text_col, label_col):
        self.df = df.reset_index(drop=True)
        self.text_col = text_col
        self.label_col = label_col
        set_of_classes = set(self.df[label_col])
        self.classes = {}
        for i, classname in enumerate(set_of_classes):
            self.classes[classname] = i

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        class_idx = self.classes[self.df.loc[idx, self.label_col]]
        return self.df.loc[idx, self.text_col], torch.tensor(class_idx, dtype=torch.float)