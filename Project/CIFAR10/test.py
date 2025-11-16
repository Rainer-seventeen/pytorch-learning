# 测试模型，将数据输入训练好的模型

import torch
import torchvision
from PIL import Image
from torchvision import transforms
from network import Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    img_path = '/root/TestPytorch/dataset/my_test_data/cat1.jpg'
    img = Image.open(img_path)
    # img = img.convert('RGB')    # 转化通道为RGB，PNG需要，JPG不用

    transform = transforms.Compose([transforms.Resize([32, 32]),
                                    transforms.ToTensor()])

    img = transform(img) # 完成模型转化
    img = torch.reshape(img, (1, 3, 32, 32))
    img = img.to(device)


    # 加载模型
    model = Model()
    # 加载模型参数（state_dict）
    model.load_state_dict(torch.load('/root/TestPytorch/models/model03231023/model49.pth', map_location=device))
    print(model)

    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(img)
    print(output)

    # CIFAR10 标准数据集
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print(classes[output.argmax(1)])

