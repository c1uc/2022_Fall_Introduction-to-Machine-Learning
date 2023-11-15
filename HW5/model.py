import torch.nn as nn


class Model(nn.Module):
    def __init__(self, task_num, device):
        super(Model, self).__init__()
        sizes = [0, 5184, 5184, 6912]
        chars = [0, 1, 2, 4]
        self.output_len = chars[task_num]
        self.device = device
        self.net = nn.Sequential(
            # 72*72 -> 68*68 or 96*72 -> 92*68
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), padding='same'),
            nn.ReLU(),
            # 68*68 -> 67*67 or 92*68 -> 91*67
            nn.MaxPool2d(kernel_size=(2, 2)),
            # 67*67 -> 63*63 or 91*67 -> 87*63
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(5, 5), padding='same'),
            nn.ReLU(),
            # 63*63 -> 62*62 or 87*63 -> 86*62
            nn.MaxPool2d(kernel_size=(2, 2)),
            # 62*62 -> 58*58 or 86*62 -> 82*58
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(5, 5), padding='same'),
            nn.ReLU(),
            # 58*58 -> 57*57 or 82*58 -> 81*57
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(p=0.3),
            # 57*57*64 -> 207936 or 81*57*64 -> 295488
            nn.Flatten(),
            nn.Linear(in_features=sizes[task_num], out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.output1 = nn.Sequential(
            nn.Linear(in_features=512, out_features=36),
            nn.Softmax(dim=0)
        )
        if self.output_len > 1:
            self.output2 = nn.Sequential(
                nn.Linear(in_features=512, out_features=36),
                nn.Softmax(dim=0)
            )
        if self.output_len > 2:
            self.output3 = nn.Sequential(
                nn.Linear(in_features=512, out_features=36),
                nn.Softmax(dim=0)
            )
            self.output4 = nn.Sequential(
                nn.Linear(in_features=512, out_features=36),
                nn.Softmax(dim=0)
            )

    def forward(self, images):
        res = []
        images = images.permute(0, 3, 1, 2)
        x = self.net(images)
        res.append(self.output1(x))
        if self.output_len > 1:
            res.append(self.output2(x))
        if self.output_len > 2:
            res.append(self.output3(x))
            res.append(self.output4(x))
        return res
