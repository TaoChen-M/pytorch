import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

# create data
n_data = torch.ones(100, 2)  # 输出100行两列的值全部都是1的张量
# 类别1的数据
x0 = torch.normal(2 * n_data, 1)  # 输出一个shape为[100,2]的矩阵，均值是2*n_data,标准差是1
# 类别1的表签
y0 = torch.zeros(100)
# 类别2的数据
x1 = torch.normal(-2 * n_data, 1)
# 类别2的标签
y1 = torch.ones(100)

# print(2*n_data)
# print(y0)
# print(-2*n_data)
# print(y1)
# plt.hist(x0)
# plt.show()


x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)


# 构建网络
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_out):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x


net = Net(n_feature=2, n_hidden=10, n_out=2)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()

for i in range(100):
    out = net.forward(x)
    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 2 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
