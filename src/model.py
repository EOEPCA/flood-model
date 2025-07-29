"""Contains a SimpleUNet model."""

import torch


class SimpleUNet(torch.nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()

        self.enc1 = torch.nn.Sequential(
            torch.nn.Conv2d(2, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
        )
        self.enc2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
        )
        self.enc3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
        )

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.dec1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
        )
        self.dec2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
        )
        self.dec3 = torch.nn.Conv2d(64, 2, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        # Decoder
        x = self.dec1(x3)
        x = self.dec2(x)
        x = self.dec3(x)

        return x

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)


def load_model(path):
    model = SimpleUNet()
    model.load_state_dict(torch.load(path))
    return model
