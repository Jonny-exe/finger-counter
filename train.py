from net import Net
import numpy as np
import argparse
import torch
import torch.nn as nn

# Loss and optimizier
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class DatasetValueTest(Dataset):
    def __init__(self, data):
        print(type(data))
        self.X = data["arr_0"][-20:]
        self.Y = data["arr_1"][-20:]

        print(type(self.X), type(self.Y), type(self.X.shape))
        print(self.X)
        print(self.Y)
        print(len(self.X), len(self.Y))
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

class DatasetValue(Dataset):
    def __init__(self, data):
        print(type(data))
        self.X = data["arr_0"][:-20]
        self.Y = data["arr_1"][:-20]

        print(type(self.X), type(self.Y), type(self.X.shape))
        print(self.X)
        print(self.Y)
        print(len(self.X), len(self.Y))
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

def train(dataset=[]):
    DEVICE = "cuda"
    EPOCHS = 100
    BATCH_SIZE = 32

    model = Net()
    if DEVICE == "cuda":
        model.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    print(dataset)
    trainsetXY = DatasetValue(dataset)
    train_loader = DataLoader(trainsetXY, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        print(f"{epoch + 1} / {EPOCHS}")
        # for i, (data, target) in enumerate(train_loader, 0):
        for (data, target) in train_loader:
            target = target.unsqueeze(-1)
            if DEVICE == "cuda":
                data, target = data.to(DEVICE), target.to(DEVICE)
            data, target = data.to(torch.float32), target.to(torch.float32)

            optimizer.zero_grad()
            data = data.reshape([BATCH_SIZE, 1, 50, 50])
            outputs = model(data)
            outputs = outputs.reshape([BATCH_SIZE, 5, 1])

            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # running_loss += loss.time()

            # if i % 2000 == 1999:
            # print('[%d, %5d] loss: %.3f' %
            # (epoch + 1, i + 1, running_loss / 2000))
            # running_loss = 0.0
        print(loss)
    torch.save(model.state_dict(), "models/value_new.pth")
    print("Finished training")


if __name__ == "__main__":
    data = np.load("images/data.npz", allow_pickle=True)
    train(data)

