import torch
import torch.utils.tensorboard as tb

from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear


class NetworkDemo(nn.Module):
    """
    自行搭建的网络，面向 CIFAR10 数据集
    """    
    def __init__(self):
        super(NetworkDemo, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5,
                            padding=2, stride=1)
        self.maxpool1 = MaxPool2d(kernel_size=2)
        self.conv2 = Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = MaxPool2d(kernel_size=2)
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(2)
        # 展开数据
        self.flatten = Flatten()
        # 线性
        self.linear1 = Linear(in_features=1024, out_features=64)
        self.linear2 = Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)

        return x

if __name__ == "__main__":
    net = NetworkDemo()
    print(net)
    input = torch.ones((64, 3, 32, 32))
    output = net(input)
    print(output.shape)

    # 使用 tb 来查看网络
    writter = tb.SummaryWriter()
    writter.add_graph(net, input)
    writter.close()