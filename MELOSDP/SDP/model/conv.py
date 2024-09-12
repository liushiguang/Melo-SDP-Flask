from torch import nn
import torch

class SDPClassifierCNN(nn.Module):
    def __init__(self):
        super(SDPClassifierCNN, self).__init__()
        # 第一层卷积层 + 批量归一化 + 最大池化
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 第二层卷积层 + 批量归一化 + 最大池化
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 第三层卷积层 + 批量归一化
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        # 全连接层
        self.fc1 = nn.Linear(256 * 9, 128)  # 256个通道，每个通道9个特征
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度

        # 第一层卷积、批量归一化、激活和池化
        x = nn.functional.gelu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # 第二层卷积、批量归一化、激活和池化
        x = nn.functional.gelu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # 第三层卷积、批量归一化和激活
        x = nn.functional.gelu(self.bn3(self.conv3(x)))

        # 展平成一维
        x = x.view(x.size(0), -1)

        x = nn.functional.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
