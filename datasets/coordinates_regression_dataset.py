from torch.utils.data import DataLoader, Dataset

import torch


__all__ = ["CoordinatesRegressionDataset"]

class CoordinatesRegressionDataset(Dataset):
    def __init__(self, df, text_col, latitude_col='latitude', longitude_col='longitude'):
        self.df = df.reset_index(drop=True)
        self.text_col = text_col
        self.latitude_col = latitude_col
        self.longitude_col = longitude_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        coords = (torch.tensor(self.df.loc[idx, self.latitude_col], dtype=torch.float32), \
                  torch.tensor(self.df.loc[idx, self.longitude_col], dtype=torch.float32))
        return self.df.loc[idx, self.text_col], coords


