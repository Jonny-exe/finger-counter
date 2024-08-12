import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from test import test
from net import Net
from data import DatasetValue

def train(dataset=[], test_dataset=None):
    DEVICE = "cuda"
    EPOCHS = 250
    BATCH_SIZE = 64
    BATCH_SIZE_CURRENT = 256
    EARLY_STOP = True
    EARLY_STOP_ACC = 0.87

    # BEST YET: 100 EPOCHS; BATCH 32: 0.0001 LR NO DROPOUT
    # WITHOUT DROPOUT, 30 EPOCHS, BATCH SIZE 256, LR: 0.0008
    # WITH    DROPOUT, 50 EPOCHS, BATCH SIZE 256, LR: 0.0008 --> WAY BETTER, WORSE LOSS BUT BETTER RESULTS, NO OVERFITTING
    # ON CONV LAYERS


    # NEW DATA
    # WORKS BAD BUT WORKS, 30 EPOCHS, 32 BATCH, 0.0002 LR
    # BETTER, 30 EPOCHS  , 32 BATCH, 0.0001 LR A BIT BETTER
    # WORSE   , 30 EPOCHS, 32 BATCH, 0.00008 LR A BIT BETTER


    model = Net()
    model.train()
    if DEVICE == "cuda":
        model.cuda()
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0009, momentum=0.7)
    # optimizer = optim.Adam(model.parameters(), lr=0.007)
    
    print(dataset)
    trainsetXY = DatasetValue(dataset)
    train_loader = DataLoader(trainsetXY, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    for epoch in range(EPOCHS):
        # print(optimizer.state_dict())
        running_loss = 0.0
        print(f"{epoch + 1} / {EPOCHS}")
        # for i, (data, target) in enumerate(train_loader, 0):
        for (data, target) in train_loader:
            target = target.unsqueeze(-1)
            if DEVICE == "cuda":
                data, target = data.to(DEVICE), target.to(DEVICE)
            data, target = data.to(torch.float32), target.to(torch.float32)

            BATCH_SIZE_CURRENT = data.shape[0]


            # print(data[0])
            optimizer.zero_grad()
            data = data.reshape([BATCH_SIZE_CURRENT, 1, 50, 50])

            # data = data + (0.1**0.5)*torch.randn([BATCH_SIZE_CURRENT, 1, 50, 50]) 
            # data = data + (0.5**0.5)*torch.randn([BATCH_SIZE_CURRENT, 1, 50, 50]).to(DEVICE)
            # data = data + (0.25**0.5)*torch.randn([BATCH_SIZE_CURRENT, 1, 50, 50]).to(DEVICE)
            # target = target + (0.0005**0.5)*torch.randn([BATCH_SIZE_CURRENT, 5, 1]).to(DEVICE)
            outputs = model(data)
            outputs = outputs.reshape([BATCH_SIZE_CURRENT, 5, 1])
            # print(outputs.squeeze().shape)
            # print("sum: ", outputs[0].sum())

            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()


            # running_loss += loss.time()

            # if i % 2000 == 1999:
            # print('[%d, %5d] loss: %.3f' %
            # (epoch + 1, i + 1, running_loss / 2000))
            # running_loss = 0.0
        print(loss)

        acc = test(model, test_dataset)
        model.train()
        if EARLY_STOP and acc > EARLY_STOP_ACC:
            break

        print("Train: ", model.training)
    torch.save(model.state_dict(), "models/value_new.pth")
    print("Finished training")


if __name__ == "__main__":
    train_dataset = np.load("images/data.npz", allow_pickle=True)
    test_dataset = np.load("images/test.npz", allow_pickle=True)
    train(train_dataset, test_dataset=test_dataset)

