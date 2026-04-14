import torch
import torch.nn as nn

class BasicBlock(nn.Module):

    def __init__(self, ch_in, ch_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=ch_out)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=ch_out)
        if stride == 1 and ch_in == ch_out:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(num_features=ch_out))

    def forward(self, x):
        x_sh = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + x_sh
        x = self.relu(x)
        return x

class ResNet8(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu = nn.ReLU(inplace=False)
        self.ch_current = 16

        self.layer1 = self._make_stage(ch_out=16, n_blocks=1, stride=1)
        self.layer2 = self._make_stage(ch_out=32, n_blocks=1, stride=2)
        self.layer3 = self._make_stage(ch_out=64, n_blocks=1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def _make_stage(self, ch_out, n_blocks, stride):
        blocks = []
        blocks.append(BasicBlock(self.ch_current, ch_out, stride=stride))
        for _ in range(n_blocks - 1):
            blocks.append(BasicBlock(ch_out, ch_out, stride=1))
        self.ch_current = ch_out
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
