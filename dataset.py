# 批量读取数据的一个实例，可以通过重写成员函数来实现自定义功能

from torch.utils.data import Dataset
from PIL import Image
import os

class Hymenoptera(Dataset):

    def __init__(self, root: str, label: str):
        """数据的初始化工作

        Args:
            root (str): 数据的根路径
            label (str): 某一个子文件夹的名称，也就是label
        """
        # 这里是根据不同的数据集来更改的
        # 以 hymenoptera_data 数据作为例子， label 下存放了所有的数据图片
        self.root_dir = root # root 路径
        self.label_dir = label # 标签信息
        self.img_path = os.path.join(self.root_dir, self.label_dir) # root 路径地址拼接
        self.img_list = os.listdir(self.img_path)


    def __getitem__(self, index: int):
        """获取某一个数据的具体信息

        Args:
            index (int): 数据索引号

        Returns:
            img, label
        """        
        img_path = os.path.join(self.path, self.img_list[index]) # 链接路径，获取当前的图片
        img = Image.open(img_path) # 读取到的PTL格式的图片数据
        label = self.label_dir # 文件夹名称已经全都对应这个label
        return img, label


    def __len__(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return len(self.img_list)


if __name__ == '__main__':
    data_root = '/Users/rainer/MyFiles/Code/Learning/Pytorch/dataset/hymenoptera_data'
    ant_data = Hymenoptera(root=data_root, label='ants')
    bee_data = Hymenoptera(root=data_root, label='bees')
    train_dataset = ant_data + bee_data # 两个数据集的总和
    print(len(ant_data), len(bee_data))
    print(len(train_dataset))

