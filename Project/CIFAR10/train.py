# 模拟训练的过程的文件
from network import Model

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 保存路径以及时间相关设置
current_time = datetime.now().strftime("%m%d%H%M")  # 例如 03231020

# 定义模型保存的文件夹路径
save_dir = f"/root/TestPytorch/models/model{current_time}"

# 如果文件夹不存在，则创建文件夹
os.makedirs(save_dir, exist_ok=True)


if __name__ == '__main__':
    # EX: Tensor Board
    writer = SummaryWriter("/root/TestPytorch/logs")

    # 1.准备数据集
    train_data = torchvision.datasets.CIFAR10(root='/root/TestPytorch/dataset', train=True, download=True,
                                              transform=transforms.ToTensor())
    test_data = torchvision.datasets.CIFAR10(root='/root/TestPytorch/dataset', train=False, download=True,
                                             transform=transforms.ToTensor())
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print(f"训练数据集长度:{train_data_size}")
    print(f"测试数据集长度:{test_data_size}")

    # 2.加载数据
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    # 3.搭建神经网络
    # 从 network.py 中导入
    model = Model().to(device)  # 将模型移动到GPU

    # 4.损失函数
    loss_fn = torch.nn.CrossEntropyLoss()  # 交叉熵

    # 5.优化器
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 6.设置其他训练的参数
    total_train_step = 0  # 训练次数
    total_test_step = 0  # 测试次数
    epochs = 50  # 训练次数

    # 7. 开始训练
    for epoch in range(epochs):
        print(f"——————第 {epoch + 1} 轮训练开始——————")

        # 训练部分
        model.train()  # 让网络进入训练模式，仅对部分层生效
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)  # 计算loss值

            # 优化器优化模型
            optimizer.zero_grad()  # 清空梯度值
            loss.backward()  # 反向传播，计算梯度值
            optimizer.step()  # 更新参数

            # 其他参数
            total_train_step += 1
            if total_train_step % 100 == 0:  # 每100print一次
                print(f"第 {total_train_step} 训练，Loss：{loss.item()}")
                writer.add_scalar('train loss', loss.item(), total_train_step)  # 写入loss

        # 测试部分
        model.eval()  # 验证模式，仅对部分层生效
        total_test_loss = 0.0
        total_accuracy = 0.0
        with torch.no_grad():  # 表示在测试过程中不许修改梯度值
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)  # 计算当前误差loss
                total_test_loss += loss.item()  # 累加loss
                accuracy = (outputs.argmax(1) == labels).sum().item()
                total_accuracy += accuracy

        total_test_step += 1

        avg_test_loss = total_test_loss / len(test_loader)
        print(f"整体测试Loss: {avg_test_loss}")
        writer.add_scalar('test loss', avg_test_loss, total_test_step)  # 写入loss

        avg_accuracy = total_accuracy / test_data_size
        print(f"整体测试Accuracy: {avg_accuracy}")
        writer.add_scalar('test accuracy', avg_accuracy, total_test_step)  # 写入loss

        # 保存当前轮的模型
        torch.save(model.state_dict(), os.path.join(save_dir, f"model{epoch}.pth"))
        print(f"模型已经保存已保存到 {save_dir}/model{epoch}.pth")