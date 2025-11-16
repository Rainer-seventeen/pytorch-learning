# 使用预训练好的模型,以imagenet为例子
# 主要是如何在已经训练完成的模型上更改内容
# https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageNet.html#torchvision.datasets.ImageNet



import torch
import torchvision
from conda.exports import download

if __name__ == "__main__":
    # train_data = torchvision.datasets.ImageNet(root='/root/TestPytorch/dataset', split='train', download=True,
    #                                            transform=torchvision.transforms.ToTensor())
    # 这个模型太大了，下不下来
    vgg16_true = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT) # 拿到下载好的数据集
    print(vgg16_true)
    vgg16_true.add_module('add_linear', torch.nn.Linear(1000, 10)) # 创建一个新的大类
    # vgg16_true.classifier.add_module('7', torch.nn.Linear(1000, 10)) # 在已有的大类下添加一些新的东西
    vgg16_true.classifier[6] = torch.nn.Linear(1000, 10)  # 在已有的大类下添加一些新的东西
    print(vgg16_true)