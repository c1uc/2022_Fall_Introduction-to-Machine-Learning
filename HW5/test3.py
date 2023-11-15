from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.efficientnet import EfficientNet
from torchvision.io import read_image
import torch.nn as nn
import torch
import cv2
import numpy as np


class Model(nn.Module):
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.net.conv1 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        self.net.fc = nn.Linear(in_features=512, out_features=10, bias=True)

    def forward(self, x):
        x = self.net(x)
        return x


weights = ResNet18_Weights.DEFAULT
model = resnet18()
model = Model(model)
model = model.to('cuda')

# print(model)
model.eval()
img = cv2.imread('./test/task3/0AWNnXyuce8j9KSZ.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

_, rt = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
rt = cv2.bitwise_not(rt)
kernel = np.ones((3, 2), np.uint8)

img2 = cv2.erode(rt, kernel, iterations=1)
img2 = cv2.dilate(img2, kernel, iterations=1)
img2 = cv2.bitwise_not(img2)
gray = np.expand_dims(gray, 2)

res = np.dstack((img1, img2))
res = torch.FloatTensor(res).to('cuda')
res = res.unsqueeze(0)
res = res.permute(0, 3, 1, 2)

# Step 4: Use the model and print the predicted category
prediction = model(res)
print(prediction)
