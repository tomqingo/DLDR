import torch
import torch.nn as nn

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ConvNet(nn.Module):

    def __init__(self, num_classes=10, x_dim=3, hid_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim)
        )
        self.classifier = nn.Linear(2*2*hid_dim, num_classes)
        #self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

def convnet(num_classes=10):
    return ConvNet(num_classes=num_classes)

 


 