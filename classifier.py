# MNIST 手写数字识别
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data

# 超参数定义
BATCH_SIZE = 512
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用cuda
# train Data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('mnist_data', train=True, download=True,
                   transform=transforms.Compose(
                       [transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))]
                   )),
    batch_size=BATCH_SIZE, shuffle=True)

# test data
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('mnist_data', train=False, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)


# 卷积网络定义
class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        # batch*1*28*28 每次会送入batch个样本，通道数是1（黑白图像），大小是28*28
        self.conv1 = nn.Conv2d(1, 10, 5)
        # 输入通道是1，输出通道10，卷积核5
        self.conv2 = nn.Conv2d(10, 20, 3)
        # input10,output20,conv3

        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)
        # in_size=512  输入的x 可以看成是512*1*28*28的张量
        out = self.conv1(x)  # batch*1*28*28--->batch*10*24*24
        out = F.relu(out)
        out = F.max_pool2d(out, 2)  # batch*10*24*24--->batch*10*12*12
        out = self.conv2(out)  # batch*10*12*12--->batch*20*10*10
        out = F.relu(out)
        out = out.view(in_size, -1)  # batch*20*10*10--->batch*2000  (-1  说明是自动推算，本例中中是20*10*10)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)  # 计算log(softmax(x))
        return out

model = convNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())

# define train function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('train epoch:{} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()
            ))

# define test function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set:Average loss:{:.4f},Accuarcy:{}/{}({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))

# training
for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)