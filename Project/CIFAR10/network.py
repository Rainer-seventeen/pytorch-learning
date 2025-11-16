# 训练 CIFAR10 数据集模型
# 训练流程图可以参考
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding="same")
        # self.pool1 = nn.MaxPool2d(kernel_size=2)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding="same")
        # self.pool2 = nn.MaxPool2d(kernel_size=2)
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding="same")
        # self.pool3 = nn.MaxPool2d(kernel_size=2)
        # self.flatten = nn.Flatten()
        # self.linear1 = nn.Linear(in_features=1024, out_features=64)
        # self.linear2 = nn.Linear(in_features=64, out_features=10)

        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=64),
        nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


if __name__ == '__main__':
    # 此处一般用于测试网络的输出正确性
    model = Model()
    # 下面是用于检测网络正确性的代码
    input = torch.ones(64, 3, 32, 32) # 创建全是1的数据
    output = model(input)
    print(output.size())






