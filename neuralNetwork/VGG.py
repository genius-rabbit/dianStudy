import torch
import os
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data.dataloader as Data


train_data = torchvision.datasets.CIFAR10(
    '/home/geniusrabbit/PycharmProjects/TestPyTF/cifar10', train=True, transform=transforms.ToTensor()
)

test_data = torchvision.datasets.CIFAR10(
    '/home/geniusrabbit/PycharmProjects/TestPyTF/cifar10', train=False, transform=transforms.ToTensor()
)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

train_loader = Data.DataLoader(dataset=train_data, batch_size=4, shuffle=True, num_workers=2)
test_loader = Data.DataLoader(dataset=test_data, batch_size=4, shuffle=False, num_workers=2)


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self,vgg_name):
        super(VGG, self).__init__()
        self.features = self._makelayers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _makelayers(self, cfg):
        layers = []
        inplanes =3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(inplanes, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(True)]
                inplanes = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# net = VGG('VGG11')


net = torch.load('VGGNet.pkl')
optimizer = optim.SGD(net.parameters(), lr=0.01)
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
            torch.save(net, 'VGGNet.pkl')

