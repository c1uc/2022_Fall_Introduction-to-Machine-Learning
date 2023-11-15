import csv

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Model
from dataset import TaskDataset

TRAIN_PATH = "./train"
GEN_PATH = "./gen"
TEST_PATH = "./test"
device = "cuda"

USE_GEN_DATA = False
FROM_TRAINED_MODEL = False
char_len = [0, 1, 2, 4]


def main():
    train_data = []
    epochs = [0, 50, 400, 800]

    if USE_GEN_DATA:
        with open(f'{GEN_PATH}/annotations.csv', newline='') as csvfile:
            for row in csv.reader(csvfile, delimiter=','):
                train_data.append(row)
    else:
        with open(f'{TRAIN_PATH}/annotations.csv', newline='') as csvfile:
            for row in csv.reader(csvfile, delimiter=','):
                train_data.append(row)

    for i in [1, 2, 3]:
        if USE_GEN_DATA:
            train_ds = TaskDataset(train_data, root=GEN_PATH, task=i, char_len=char_len[i])
        else:
            train_ds = TaskDataset(train_data, root=TRAIN_PATH, task=i, char_len=char_len[i])

        train_dl = DataLoader(train_ds, batch_size=500, num_workers=1, drop_last=True, shuffle=True)
        print(f"task{i} training...")

        net = Model(i, device)
        if FROM_TRAINED_MODEL:
            net.load_state_dict(torch.load(f"./models/task{i}"))
        net = net.to(device)

        # print(cnn)
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        loss_fn = nn.BCELoss()

        net.train()
        for _ in tqdm(range(epochs[i])):
            for image, label in train_dl:
                image = image.to(device)
                label = label.to(device)

                pred = net(image)
                loss = loss_fn(pred[0], label[:, 0])
                if i >= 2:
                    loss += loss_fn(pred[1], label[:, 1])
                if i == 3:
                    loss += loss_fn(pred[2], label[:, 2]) + loss_fn(pred[3], label[:, 3])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print("saving model...")
        torch.save(net.state_dict(), f"./models/task{i}")


if __name__ == '__main__':
    main()
