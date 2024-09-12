import torch.nn as nn
import torch

# 定义线性分类器
class SDPClassifierNN(nn.Module):
    def __init__(self, input_dim):
        super(SDPClassifierNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(16)
        self.fc5 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = nn.functional.gelu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = nn.functional.gelu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = nn.functional.gelu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = nn.functional.gelu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc5(x))
        return x