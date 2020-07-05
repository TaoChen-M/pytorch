# use pytorch create regression model
# create a function y=a*x^2+b and add noise to visualize it

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data(tensor) shape(100,1)
y = x.pow(2) + 0.2 * torch.rand(x.size())  # add noise


# # 画图
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

# 建立神经网络
# 先定义所有的层属性（_init_()) 然后再一层层搭建（forward（））层和层之间的关系链接
class Net(torch.nn.Module):  # 继承torch的Module
    def __init__(self, n_feature, n_hidden, n_output):
        # 定义层的信息 n_feature表示有多少个输入 n_hidden 隐藏层神经元个数  n_output输出个数

        super(Net, self).__init__()  # 继承_init_功能

        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

    def forward(self, x):
        # x是输入data 这同时也是Modul中的forward功能,定义神经网络前向传递的过程，把_init_中的层信息一个个组合起来

            # 正向传播输入值，神经网络分析输出值
            x = F.relu(self.hidden(x))  # 激励函数（隐藏层的线性值）
            x = self.predict(x)  # 输出值
            return x


net = Net(n_feature=1, n_hidden=10, n_output=1)

print(net)  # net 的结构

plt.ion()

# 训练网络
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 传入net 的所有参数
loss_fun = torch.nn.MSELoss()  # 预测值和真实值之间的误差计算公式（均方差）

for i in range(200):
    prediction = net.forward(x)  # 喂给net训练数据x,输出预测值

    loss = loss_fun(prediction, y)  # 计算两者的误差

    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward()  # 误差反向传播，计算参数更新值
    optimizer.step()  # 将参数更新值加到net的parameters

    if i % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
