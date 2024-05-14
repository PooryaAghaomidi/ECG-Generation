from torch import nn


class Fully_connected(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear(131072, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
            nn.Softmax(dim=1))

    def forward(self, x):
        x /= 0.18215

        for module in self:
            x = module(x)

        return x
