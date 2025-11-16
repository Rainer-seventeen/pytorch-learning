# TensorBoard 一个常用于训练结果可视化的工具

from torch.utils.tensorboard import SummaryWriter
# from PIL import Image
# import numpy as np
import cv2 as cv
# import torch

if __name__ == '__main__':

    writer = SummaryWriter(log_dir='/root/learn_dataset/logs') # 将事件存储到目标logs文件夹
    img_path = '/root/learn_dataset/train/ants/6240329_72c01e663e.jpg'
    img_cv = cv.imread(img_path)
    print(img_cv.shape) # HWC
    writer.add_image('test', img_cv, 0, dataformats='HWC')

    # writer.add_scalar(tag='loss', scalar_value=0.5)
    # tag 代表图表标题
    # scalar_value 代表要保存的数据
    # global_step 代表训练的总步数

    # 读取方式：在logs的父文件夹下运行 tensorboard --logdir=logs
    for i in range(100):
        writer.add_scalar(tag='loss2', scalar_value=2 * i, global_step=i)

    writer.close()
