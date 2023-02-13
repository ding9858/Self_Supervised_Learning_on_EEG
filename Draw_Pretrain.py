import os
import torch
import matplotlib.pyplot as plt
if __name__ == "__main__":
    epoch=[]
    data_file = f'Contrastive_loss_Continue.pt'
    contrastive_loss=torch.load(os.path.join(os.getcwd(), data_file), map_location=torch.device('cpu'))
    for i in range(len(contrastive_loss)):
        epoch.append(i)

    plt.figure()
    plt.plot(epoch, contrastive_loss, color='g')
    plt.savefig('Test.png')
    plt.show()