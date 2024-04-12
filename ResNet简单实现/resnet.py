import torch
from torch import nn
from torch.nn import functional as F

# 每个模块包含4个卷积层
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                    kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                    kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                        kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        
        self.bn1 = self.BatchNorm2d(num_channels)
        self.bn2 = self.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

"""
ResNet使用了四个有残差块组成的模块，每个模块使用若干个同样输出通道数的残差块。
第一个模块的通道数同输入通道数一致。由于Resnet开头使用了步幅为2的最大汇聚层，所以无需减少高和宽。
之后的每个模块再第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半。
"""

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2)) # 如果不是第一个模块，那么第一个残差块需要将长宽减半
        else:
            blk.append(Residual(num_channels, num_channels)) # 否者输入的图像的长宽不变，通道数也不变
    return blk

# 这里的18层只计算了3x3卷积、7x7卷积和最后的全连接层。
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2(3, 64, kernel_size=7, stride=2, padding=3),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))

        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.ft = nn.Flatten()
        self.fc = nn.Linear(512, 10)
    
    def forward(self, X):
        X = self.b5(self.b4(self.b3(self.b2(self.b1(X)))))
        X = self.avg(X)
        X = self.ft(X)
        X = self.fc(X)
        return X