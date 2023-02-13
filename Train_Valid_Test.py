import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
import numpy as np
from Dataset_train import EEGDataset
from torchvision import models
import matplotlib.pyplot as plt

def train(dataloader, model, criterion, optimizer):
    model.train()
    train_acc = 0.0
    train_loss = 0.0
    for X, y in dataloader:
        y_pred = model(X)

        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("-------------------------------")
        print("In training,  Loss for this Batch: ", loss)
        print("-------------------------------")
        train_loss += loss.item()
        train_acc += (y_pred.argmax(dim=1) == y).sum()

    nb_samples = len(dataloader.dataset)
    nb_batches = len(dataloader)

    train_acc /= nb_samples
    train_loss /= nb_batches
    print(f"Train loss for epoch : {train_loss:>8f}")
    print(f"Train acc for epoch :  {train_acc:>8f}")

    return train_acc, train_loss

def valid(dataloader, model, criterion, optimizer):
    model.eval()
    valid_acc  = 0.0
    valid_loss = 0.0
    for X, y in dataloader:
        y_pred = model(X)

        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        valid_loss+= loss.item()
        valid_acc += (y_pred.argmax(dim=1) == y).sum()

    nb_samples = len(dataloader.dataset)
    nb_batches = len(dataloader)

    valid_acc /= nb_samples
    valid_loss /= nb_batches
    print(f"Valid loss for epoch : {valid_loss:>8f}")
    print(f"Valid acc for epoch :  {valid_acc:>8f}")
    return valid_acc, valid_loss

def test(dataloader, model, criterion, optimizer):
    model.eval()
    Test_acc  = 0.0
    Test_loss = 0.0
    for X, y in dataloader:
        y_pred = model(X)

        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Test_loss+= loss.item()
        Test_acc += (y_pred.argmax(dim=1) == y).sum()

    nb_samples = len(dataloader.dataset)
    nb_batches = len(dataloader)

    Test_acc /= nb_samples
    Test_loss /= nb_batches
    print(f"Test loss  : {Test_loss:>8f}")
    print(f"Test acc   : {Test_acc:>8f}")
    return Test_acc, Test_loss

def change_layers(model):
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(512, 512, bias=True)
    return model

def early_stopping(min_loss,loss_list):
    if (loss_list[-1] > min_loss
            and loss_list[-2] > min_loss
            and loss_list[-3] > min_loss
            and loss_list[-4] > min_loss
            and loss_list[-5] > min_loss
            and loss_list[-6] > min_loss
            and loss_list[-7] > min_loss
            and loss_list[-8] > min_loss
            and loss_list[-9] > min_loss
    ):
        return True

def Draw_acc_Loss(epoch, train_acc_list,train_loss_list,valid_loss_list,valid_acc_list):
    plt.figure()
    plt.plot(epoch, train_loss_list, color='g')
    plt.plot(epoch, valid_loss_list, color='r')
    plt.savefig('Loss.png')
    plt.show()

    plt.plot(epoch, train_acc_list, color='g')
    plt.plot(epoch, valid_acc_list, color='r')
    plt.savefig('Acc.png')
    plt.show()

if __name__ == "__main__":
    dataset = EEGDataset(augment=True)
    train_len = int(0.7 * len(dataset))
    validation_len = int(0.15 * len(dataset))
    test_len = len(dataset) - train_len - validation_len

    train_data, test_data, validation_data = random_split(dataset, [train_len, test_len, validation_len])

    train_dataloader = DataLoader(train_data, batch_size=10, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=5, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(validation_data, batch_size=5, shuffle=True, drop_last=True)

    model = models.resnet18(pretrained=True)
    model = change_layers(model)
    model.load_state_dict(torch.load('Pretrain_Continue.pth'))
    model_train = nn.Sequential(
        model,
        nn.Linear(512, 2, bias=True)
    )

    optimizer = torch.optim.Adam(model_train.parameters(), lr=3e-2, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)


    train_acc_list = []
    train_loss_list= []
    valid_acc_list = []
    valid_loss_list= []

    min_loss = np.inf

    for t in range(150):
        print("*******Epoch: ", t)
        train_acc, train_loss = train(train_dataloader, model_train, criterion, optimizer)
        valid_acc, valid_loss = valid(valid_dataloader, model_train, criterion, optimizer)

        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        valid_acc_list.append(valid_acc)
        valid_loss_list.append(valid_loss)


        if min_loss > valid_loss:
            print("valid loss decreased!Saving The Model...")
            min_loss = valid_loss
            torch.save(model_train.state_dict(), 'Train_Best_Valid.pth')
            print("Model updated!")

        torch.save(model_train.state_dict(), 'Train.pth')

        torch.save(valid_loss_list, 'Valid_loss.pt')
        torch.save(valid_acc_list, 'Valid_acc.pt')
        torch.save(train_loss_list, 'train_loss.pt')
        torch.save(train_acc_list, 'train_acc.pt')






