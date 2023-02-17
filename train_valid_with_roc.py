import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
import numpy as np
from Dataset_train import EEGDataset
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
def train(dataloader, model, criterion, optimizer):
    model.train()
    train_acc = 0.0
    train_loss = 0.0
    num_classes = 2
    y_list = [[]]  # first dim is original
    y_pred_list = []
    roc_auc = []
    fprtpr = []  # first dim is classes, second dim is fpr and tpr
    for clas_ in range(num_classes):
        y_list.append([])
        y_pred_list.append([])  # [[], []]
        roc_auc.append([])  # [[], []]
        fprtpr.append([])  # first dim is classes, second dim is fpr and tpr
        fprtpr[-1].append([])  # first dim is classes, second dim is fpr and tpr
        fprtpr[-1].append([])  # first dim is classes, second dim is fpr and tpr [[[], []], [[], []]]

    for X, y in dataloader:
        y_pred1 = model(X)

        loss = criterion(y_pred1, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (y_pred1.argmax(dim=1) == y).sum()
        y_list[0].append(int(y[0]) + 1)
        for class_ in range(num_classes):
            y_pred_list[class_].append(float(y_pred1[0][class_]))# append y_pred
        test_ = False

        if test_ == True and 1 in y_list[0] and 2 in y_list[0] and 3 in y_list[0] and 4 in y_list[0]: # if test model: only process few samples
            for class_ in range(num_classes):
                y_list[int(class_) + 1] = list(map(lambda x: 1 if x != class_ + 1 else 0, y_list[0]))#turn every element that is not target to 1 since roc can only process binary class
                print("y_list" + str(int(class_) + 1) + "=", y_list[int(class_) + 1])
                print("y_pred_list" + str(class_) + "=", y_pred_list[class_])
                fprtpr[class_][0], fprtpr[class_][1], thersholds = roc_curve(y_list[int(class_) + 1],
                                                                             y_pred_list[class_],
                                                                             pos_label=0)
                roc_auc[class_].append(auc(fprtpr[class_][0], fprtpr[class_][1]))

            nb_samples = len(dataloader.dataset)
            nb_batches = len(dataloader)

            train_acc /= nb_samples
            train_loss /= nb_batches
            return train_acc, train_loss, roc_auc, fprtpr


    print("y_list_origin=", y_list[0])
    for class_ in range(num_classes):
        y_list[int(class_) + 1] = list(map(lambda x: 1 if x != class_ + 1 else 0, y_list[0]))
        print("y_list" + str(int(class_) + 1) + "=", y_list[int(class_) + 1])
        print("y_pred_list" + str(class_) + "=", y_pred_list[class_])
        fprtpr[class_][0], fprtpr[class_][1], thersholds = roc_curve(y_list[int(class_) + 1], y_pred_list[class_],
                                                                     pos_label=0)
        roc_auc[class_].append(auc(fprtpr[class_][0], fprtpr[class_][1]))

    nb_samples = len(dataloader.dataset)
    nb_batches = len(dataloader)

    train_acc /= nb_samples
    train_loss /= nb_batches
    return train_acc, train_loss, roc_auc, fprtpr

def valid(dataloader, model, criterion, optimizer):
    model.eval()
    valid_acc  = 0.0
    valid_loss = 0.0
    y_validlabel_list = []
    y_valid_list = []
    num_classes = 2
    for clas_ in range(num_classes):
        y_valid_list.append([])

    for X, y in dataloader:
            y_pred1 = model(X)
            loss = criterion(y_pred1, y)
            valid_acc += (y_pred1.argmax(dim=1) == y).sum()
            valid_loss += loss.item()
            y_validlabel_list.append(int(y[0]) + 1)
            for class_ in range(num_classes):
                y_valid_list[class_].append(float(y_pred1[0][class_]))
    print("y_validlabel_list=", y_validlabel_list)
    for class_ in range(num_classes):
        print("y_valid_list" + str(class_)+"="+str(y_valid_list[class_]))

    nb_samples = len(dataloader.dataset)
    nb_batches = len(dataloader)
    valid_acc /= nb_samples
    valid_loss /= nb_batches
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

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

    model = models.resnet18(pretrained=True)
    model = change_layers(model)
    model.load_state_dict(torch.load('Pretrain_Continue.pth'))
    model_train = nn.Sequential(
        model,
        nn.Linear(512, 2, bias=True)
    )
    #model_train.load_state_dict(torch.load('Train_Best_Valid.pth'))

    optimizer = torch.optim.Adam(model_train.parameters(), lr=3e-2, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    epochs = []

    train_acc_list = []
    train_loss_list= []
    valid_acc_list = []
    valid_loss_list= []

    min_loss = np.inf
    num_classes = 2
    auc_list = []

    for classes_ in range(num_classes):
        auc_list.append([])

    for t in range(100):
        print(f"Epoch {t + 1}\n-------------------------------")
        roc_coordinate=[]
        train_acc, train_loss,roc_auc,fprtpr = train(train_dataloader, model_train, criterion, optimizer)
        valid_acc, valid_loss = valid(valid_dataloader, model_train, criterion, optimizer)
        epochs.append(t)

        roc_coordinate.append(fprtpr)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        valid_acc_list.append(valid_acc)
        valid_loss_list.append(valid_loss)
        print("|Train Loss: ", train_loss, "|  &   |Train Acc: ", train_acc)
        print("|Valid Loss: ", valid_loss, "|  &   |Valid Acc: ", valid_acc)
        for classes_ in range(len(fprtpr)):
            auc_list[classes_].append(roc_auc[classes_])
            # print("auc_new" + str(classes_) + "=", roc_auc[classes_])
            print("auc_list" + str(classes_) + "=", auc_list[classes_])
            #print("roc_coordinate" + str(classes_) + "=", fprtpr[classes_])

        print("train_loss_list=", train_loss_list)
        print("train_acc_list=", train_acc_list)
        print("valid_acc_list=", valid_acc_list)
        print("valid_loss_list=", valid_loss_list)


        if min_loss > valid_loss:
            print("valid loss decreased!Saving The Model...")
            min_loss = valid_loss
            torch.save(model.state_dict(), 'Train.pth')
            print("Model updated!")



        torch.save(valid_loss_list, 'Valid_loss.pt')
        torch.save(valid_acc_list, 'Valid_acc.pt')
        torch.save(train_loss, 'train_loss.pt')
        torch.save(train_acc, 'train_acc.pt')
    print("auc_list", auc_list, "train_loss_list", train_loss_list,
              "train_acc_list", train_acc_list, "valid_acc_list", valid_acc_list, "valid_loss_list", valid_loss_list)

    Draw_acc_Loss(epochs, train_acc_list,train_loss_list,valid_loss_list,valid_acc_list)




