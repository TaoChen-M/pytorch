import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input channel 6 ouput channel  5*5 convolution
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 定义该网络的向前传播函数，一旦定义好了向后传播函数就会自动生成
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果窗口是个正方形也可以直接写一个数字
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, self.num_flat_features(x))  # view函数将张量x变成一个一维的向量形式   计算特征的维数，指定特征维数之后分配行数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # 使用num_flat_features() 计算张量x的特征总量 把每一个数字看作独立的特征  比如4*2*2的张量特征数就是16
    def num_flat_features(self, x):
        size = x.size()[1:]  # 这里使用[1:] 是因为pytorch只接受批输入 一次性输入很多张图片 那么输入的数据张量维度自然上升到四维 [1:]只取后三维
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

# torch.nn只支持小批量输入  列如nn.Conv2d接受一个4维的张量，每一维分别是（样本数，通道数，高，宽）
input = torch.randn(1, 1, 32, 32)
# out = net(input)
# print(out)

# 将所有参数的梯度缓存清零 然后进行随机梯度的反向传播
# net.zero_grad()
# out.backward(torch.randn(1, 10))

# loss计算
output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
loss_func = nn.MSELoss()
loss = loss_func(output, target)
print(loss)

# 反向传播  梯度优化
# 在调用反向传播之前 需要清除已经存在的梯度，否则梯度会被累加到已存在的梯度
# 调用loss.backward()查看conv1层的偏差在反向传播前后的梯度
optimizer=optim.SGD(net.parameters(),lr=0.01)
optimizer.zero_grad()
print('conv1 bias,grad before backward')
print(net.conv1.bias.grad)
loss.backward()
print('conv1 bias.grad after backward')
print(net.conv1.bias.grad)
optimizer.step()
