import os

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

epoch=1  # train the training data n times
batch_size=50
lr=0.001   # learning rate
download_mnist=False

if not(os.path.exists('./MNIST_data/'))or not os.listdir('./MNIST_data/'):
    # not mnist dir or mnist is empty dir
    download_mnist=True

# mnist手写数字
train_data=torchvision.datasets.MNIST(
    root='./MNIST_data',   # 保存或者提取位置
    train=True,    # 表明这是训练数据
    transform=torchvision.transforms.ToTensor(),   # 将图像数据转换为tensor格式
    download=download_mnist,   # 没有下载就进行下载
)

print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
plt.title('%i'%train_data.train_labels[0])
plt.show()

train_loader=Data.Dataloader(dataset=train_data,batch_size=batch_size,shuffle=True)

# 测试集
test_data=torchvision.datasets.MNIST(root='./MNIST_data/',train=False)
test_x=torch.unsqueeze(test_data.test_data,dim=1).type(torch.FloatTensor)[:2000]/255
test_y=test_data.test_labels[:2000]

nn.MaxPool2d

