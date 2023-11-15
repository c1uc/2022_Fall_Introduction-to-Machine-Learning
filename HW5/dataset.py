import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class TaskDataset(Dataset):
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    CHAR2LABEL = {char: i for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, data, root, task, char_len, return_filename=False):
        self.data = [d for d in data if d[0].startswith(f"task{task}")]
        self.return_filename = return_filename
        self.root = root
        self.char_len = char_len

    def __getitem__(self, index):
        filename, label = self.data[index]

        img = cv2.imread(f"{self.root}/{filename}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, rt = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        rt = cv2.bitwise_not(rt)
        kernel = np.ones((3, 2), np.uint8)

        img2 = cv2.erode(rt, kernel, iterations=1)
        img2 = cv2.dilate(img2, kernel, iterations=1)
        img2 = cv2.bitwise_not(img2)
        img2 = np.expand_dims(img2, 2)
        if self.return_filename:
            return torch.FloatTensor(img2), filename
        else:
            target = np.zeros((self.char_len, len(TaskDataset.CHARS)))
            for i, l in enumerate(label):
                target[i, TaskDataset.CHAR2LABEL[l]] = 1

            target = torch.FloatTensor(target)
            return torch.FloatTensor(img2), target

    def __len__(self):
        return len(self.data)
