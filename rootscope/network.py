"""Inference architecture from the manuscript's controlled evaluation."""
import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, intermediate_channels):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(gate_channels, intermediate_channels, 1), nn.BatchNorm2d(intermediate_channels))
        self.skip = nn.Sequential(nn.Conv2d(skip_channels, intermediate_channels, 1), nn.BatchNorm2d(intermediate_channels))
        self.psi = nn.Sequential(nn.Conv2d(intermediate_channels, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, gate, skip):
        return skip * self.psi(F.relu(self.gate(gate) + self.skip(skip), inplace=True))


class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.d1, self.d2 = DoubleConv(1, 32), DoubleConv(32, 64)
        self.d3, self.d4 = DoubleConv(64, 128), DoubleConv(128, 256)
        self.bottleneck = DoubleConv(256, 512)
        self.a4, self.a3 = AttentionGate(512, 256, 128), AttentionGate(256, 128, 64)
        self.a2, self.a1 = AttentionGate(128, 64, 32), AttentionGate(64, 32, 16)
        self.u1, self.u2 = DoubleConv(768, 256), DoubleConv(384, 128)
        self.u3, self.u4 = DoubleConv(192, 64), DoubleConv(96, 32)
        self.out = nn.Conv2d(32, 1, 1)

    @staticmethod
    def up(x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        x4 = self.d4(self.pool(x3))
        b = self.bottleneck(self.pool(x4))
        z = self.up(b)
        z = self.u1(torch.cat([z, self.a4(z, x4)], dim=1))
        z3 = self.up(z)
        z = self.u2(torch.cat([z3, self.a3(z3, x3)], dim=1))
        z2 = self.up(z)
        z = self.u3(torch.cat([z2, self.a2(z2, x2)], dim=1))
        z1 = self.up(z)
        z = self.u4(torch.cat([z1, self.a1(z1, x1)], dim=1))
        return torch.sigmoid(self.out(z))
