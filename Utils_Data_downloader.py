from scipy import signal
import torch
from torch.utils.data import Dataset
from mne.datasets import eegbci
from mne.io import read_raw_edf

runs1 = 6
runs2 = 10
runs3 = 12
runs4 = 14

class Downloader(Dataset):

    def __init__(self, augment=True) -> None:
        self.augment=augment
        self.data=[]
        self.labels = []
        for i in range(100):

            files1 = eegbci.load_data(i+1, runs1, '../datasets/')
            raws1 = [read_raw_edf(f, preload=True) for f in files1]
            raw_data1= raws1[0].get_data()
            raw_data1= raw_data1[:,:15000]

            files2 = eegbci.load_data(i+1, runs2, '../datasets/')
            raws2 = [read_raw_edf(f, preload=True) for f in files2]
            raw_data2 = raws2[0].get_data()
            raw_data2 = raw_data2[:, :15000]

            files3 = eegbci.load_data(i + 1, runs3, '../datasets/')
            raws3 = [read_raw_edf(f, preload=True) for f in files3]
            raw_data3 = raws3[0].get_data()
            raw_data3 = raw_data3[:, :15000]

            files4 = eegbci.load_data(i + 1, runs4, '../datasets/')
            raws4 = [read_raw_edf(f, preload=True) for f in files4]
            raw_data4 = raws4[0].get_data()
            raw_data4= raw_data4[:, :15000]

            if self.augment == True:
                samp_freq = 200
                notch_freq = 50.0  
                quality_factor = 0.2  
                b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)

                noisySignal1 = raw_data1
                sample1 = torch.from_numpy(signal.filtfilt(b_notch, a_notch, noisySignal1).copy())

                noisySignal2 = raw_data2
                sample2 = torch.from_numpy(signal.filtfilt(b_notch, a_notch, noisySignal2).copy())

                noisySignal3 = raw_data3
                sample3 = torch.from_numpy(signal.filtfilt(b_notch, a_notch, noisySignal3).copy())

                noisySignal4 = raw_data4
                sample4 = torch.from_numpy(signal.filtfilt(b_notch, a_notch, noisySignal4).copy())

            sample1 = torch.reshape(sample1, (1, 64, 15000))
            sample2 = torch.reshape(sample2, (1, 64, 15000))
            sample3 = torch.reshape(sample3, (1, 64, 15000))
            sample4 = torch.reshape(sample4, (1, 64, 15000))

            self.data.append(sample1)
            self.labels.append(torch.tensor(0))

            self.data.append(sample2)
            self.labels.append(torch.tensor(1))

            self.data.append(sample3)
            self.labels.append(torch.tensor(2))

            self.data.append(sample4)
            self.labels.append(torch.tensor(3))

        torch.save(self.data, 'Uni4C_data.pt')
        torch.save(self.labels, 'Uni4C_label.pt')


if __name__ == "__main__":
    data=Downloader(augment=True)
