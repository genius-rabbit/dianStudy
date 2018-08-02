import os
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
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


class AlexNet(nn.Module):
    def __init__(self, num_class=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(3072, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_class)
        )

    def forward(self, inputs):
        out = self.features(inputs)
        out = inputs.view(out.size(0), -1)
        out = self.classifier(out)
        return out


net = AlexNet()
# net = torch.load('Alex.pkl')
optimizer = optim.SGD(net.parameters(), lr=0.03)
loss_func = nn.CrossEntropyLoss()

for epoch in range(50):
    train_acc1 = 0
    for i, data in enumerate(test_loader, 0):
        inputs, lables = data
        inputs, lables = Variable(inputs), Variable(lables)
        out = net(inputs)
        pred = torch.max(out, 1)[1]
        train_acc1 += (pred == lables).sum().item()
        if i % 1000 == 999:
            print('testDataAcc:[%d,%5d] acc: %.6f' % (epoch + 1, i + 1, train_acc1 / (1000 * lables.size(0))))
            train_acc1 = 0

    train_loss = 0
    train_acc = 0
    train_loss1 = 0
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

        if i % 100 == 99:
            print('[%d,%5d] loss:%.6f' % (epoch+1, i+1, train_loss/100))
            train_loss1 = train_loss1+train_loss
            train_loss = 0

        if i % 2000 == 1999:
            print('>>>>[%d,%5d] loss:%.6f' % (epoch + 1, i + 1, train_loss1 / 2000))
            print('>>>>[%d,%5d] acc: %.6f' % (epoch + 1, i + 1, train_acc / (2000 * lables.size(0))))
            train_acc = 0
            train_loss1 = 0
            torch.save(net, 'Alex0.03.pkl')

