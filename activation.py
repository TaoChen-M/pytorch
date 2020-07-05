import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# make data
x=torch.linspace(-5,5,200)
x=Variable(x)

# 转换为numpy 格式   出图的时候用
x_np=x.data.numpy()

#激励函数激活
y_relu=torch.relu(x).data.numpy()
y_sigmod=torch.sigmoid(x).data.numpy()
y_tanh=torch.tanh(x).data.numpy()
y_softplus=F.softplus(x).data.numpy()

# 绘图
plt.figure(1,figsize=(8,6))
# figure(num,figsize) num图像编号或名称 数字为编号 字符串为名称   figsize是宽和高
plt.subplot(221)
# 在一副图像上创建单个子图 221表示2*2个子图左上角第一个
plt.plot(x_np,y_relu,c='red',label='relu')  # 生成线图
plt.ylim((-1,5))  # 设定y坐标轴的范围
plt.legend(loc='best')  #设置图例位置

plt.subplot(222)
plt.plot(x_np,y_sigmod,c='red',label='sigmod')
plt.ylim((-0.2,1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np,y_tanh,c='red',label='tanh')
plt.ylim((-1.2,1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np,y_softplus,c='red',label='softplus')
plt.ylim((-0.2,6))
plt.legend(loc='best')

plt.show()