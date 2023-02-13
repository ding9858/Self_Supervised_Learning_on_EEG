import torch
import matplotlib.pyplot as plt
from Dataset_train import EEGDataset

dataset = EEGDataset(augment=True)
subject = 1
data = dataset.__getitem__(subject)

#data = torch.reshape(data, (65, 55000))
#channel = 22
x_t = data[0]
x_t = torch.reshape(x_t, (65, 55000))
x_t =x_t[1]
fs = 200
N = len(x_t)
t = torch.arange(0, N) * 1/fs

plt.figure()
plt.plot(t, x_t)
plt.xlabel('t [s]')
plt.ylabel('x(t)')
plt.savefig('EEG_recording.png')
plt.show()