import os
import torch
from scipy import signal
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    """Fast EEGDataset (fetching prepared data and labels from files)"""

    def __init__(self, augment=True) -> None:
        data_file = f'data_Dinh_fs200.pt'
        labels_file = f'labels_Dinh_fs200.pt'
        self.augment = augment
        self.data_raw= torch.load(os.path.join(os.getcwd(), data_file), map_location=torch.device('cpu'))
        self.labels_raw = torch.load(os.path.join(os.getcwd(), labels_file), map_location=torch.device('cpu'))

        self.data = []
        self.labels = []

        for i in range(len(self.data_raw)):
            sample, sample_label = self.data_raw[i], self.labels_raw[i]

            if self.augment == True:
                samp_freq = 200
                notch_freq = 50.0  # Frequency to be removed from signal (Hz)
                quality_factor = 0.2  # Quality factor
                # Design a notch filter using signal.iirnotch
                b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)
                # Compute magnitude response of the designed filter
                noisySignal = sample
                # Apply notch filter to the noisy signal using signal.filtfilt
                sample = torch.from_numpy(signal.filtfilt(b_notch, a_notch, noisySignal).copy())

            sample = torch.reshape(sample, (1, 65, 55000))

            self.data.append(sample)

            if sample_label >= 1:
                self.labels.append(torch.tensor(1))
            else:
                self.labels.append(torch.tensor(0))

    def __len__(self) -> int:
        return len(self.labels)
    def __getitem__(self, idx):
        """return a sample from the dataset at index idx"""
        data, label = self.data[idx].float(),self.labels[idx]
        return data, label

