# 演示一种用于语义分割中常用的检测模型的指标
# argmax 参数 dimension = 1 代表横向
import torch

if __name__ == '__main__':
    outputs = torch.tensor([[0.01, 0.2],
                            [0.3, 0.5]])

    preds = outputs.argmax(dim=1)
    targets = torch.tensor([[0, 1]])
    print(preds.eq(targets).sum()) # 计算和目标相等的个数