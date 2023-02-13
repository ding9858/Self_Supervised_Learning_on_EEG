import os
import torch
import matplotlib.pyplot as plt
if __name__ == "__main__":
    epoch=[]
    train_acc = f'train_acc.pt'
    train_acc_list=torch.load(os.path.join(os.getcwd(), train_acc), map_location=torch.device('cpu'))
    train_loss = f'train_loss.pt'
    train_loss_list = torch.load(os.path.join(os.getcwd(), train_loss), map_location=torch.device('cpu'))

    valid_acc = f'Valid_acc.pt'
    valid_acc_list = torch.load(os.path.join(os.getcwd(), valid_acc), map_location=torch.device('cpu'))
    valid_loss = f'Valid_loss.pt'
    valid_loss_list = torch.load(os.path.join(os.getcwd(), valid_loss), map_location=torch.device('cpu'))


    for i in range(len(valid_loss_list)):
        epoch.append(i)

    plt.figure()
    plt.plot(epoch, valid_loss_list, color='r')
    plt.savefig('Valid_loss.png')
    plt.show()

    plt.plot(epoch, valid_acc_list, color='r')
    plt.savefig('Valid_acc.png')
    plt.show()

    plt.figure()
    plt.plot(epoch, train_loss_list, color='g')
    plt.savefig('train_loss.png')
    plt.show()

    plt.plot(epoch, train_acc_list, color='g')
    plt.savefig('train_acc.png')
    plt.show()
