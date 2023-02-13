import os
import torch
from torch.utils.data import Dataset
class EEGDataset(Dataset):
    """Fast EEGDataset (fetching prepared data and labels from files)"""

    def __init__(self, augment=True) -> None:
        data_file = f'Uni4C_data.pt'
        labels_file = f'Uni4C_label.pt'
        self.augment = augment
        self.data= torch.load(os.path.join(os.getcwd(), data_file), map_location=torch.device('cpu'))
        self.labels = torch.load(os.path.join(os.getcwd(), labels_file), map_location=torch.device('cpu'))
    def __len__(self) -> int:
        return len(self.labels)
    def __getitem__(self, idx):
        """return a sample from the dataset at index idx"""
        data, label = self.data[idx].float(),self.labels[idx]
        return data, label

