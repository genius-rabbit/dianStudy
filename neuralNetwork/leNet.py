import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as Data

train_data = torchvision.datasets.CIFAR10(
    '/home/geniusrabbit/PycharmProjects/TestPyTF/cifar10', train=True, transform=transforms.ToTensor(),
)

test_data = torchvision.datasets.CIFAR10(
    '/home/geniusrabbit/PycharmProjects/TestPyTF/cifar10', train=False, transform=transforms.ToTensor()
)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

train_loader = Data.DataLoader(dataset=train_data, batch_size=4, shuffle=True, num_workers=2)
test_loader = Data.DataLoader(dataset=test_data, batch_size=4, shuffle=False, num_workers=2)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 32*32
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 14*14
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 5*5
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()

optimizer = optim.SGD(net.parameters(), lr=0.012)
loss_func = nn.CrossEntropyLoss()

for epoch in range(50):
    train_loss = 0
    train_acc = 0
    for i, data in enumerate(train_loader, 0):

        inputs, lables = data
        inputs, lables = Variable(inputs), Variable(lables)

        optimizer.zero_grad()

        out = net(inputs)
        loss = loss_func(out, lables)
        train_loss += loss.item()

        pred = torch.max(out, 1)[1]
        train_acc += (pred == lables).sum().item()

        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()

        if i % 2000 == 1999:
            print('[%d,%5d] loss:%.6f' % (epoch+1, i+1, train_loss/2000))
            print('[%d,%5d] acc: %.6f' % (epoch + 1, i + 1, train_acc / (2000*lables.size(0))))
            train_loss = 0
            train_acc = 0
    train_acc1 = 0
    for i, data in enumerate(test_loader, 0):
        inputs, lables = data
        inputs, lables = Variable(inputs), Variable(lables)
        out = net(inputs)
        pred = torch.max(out, 1)[1]
        train_acc1 += (pred == lables).sum().item()
        if i % 2000 == 1999:
            print('test:[%d,%5d] acc: %.6f' % (epoch + 1, i + 1, train_acc1 / (2000 * lables.size(0))))
            train_loss = 0
            train_acc = 0
