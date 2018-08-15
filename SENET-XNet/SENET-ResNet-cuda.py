import torch
import math
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data.dataloader as Data

print('load data...')
# root = '/home/geniusrabbit/PycharmProjects/TestPyTF/cifar100'
root = '/input/cifar'
train_data = torchvision.datasets.CIFAR100(
    root=root, train=True, transform=transforms.ToTensor()
)

test_data = torchvision.datasets.CIFAR100(
    root=root, train=False, transform=transforms.ToTensor()
)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

train_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=2)
test_loader = Data.DataLoader(dataset=test_data, batch_size=100, shuffle=False, num_workers=2)
print('load data success...')


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        self.fc1 = nn.Conv2d(planes, planes // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes // 16, planes, kernel_size=1)

    def forward(self, x):
        res = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        w = F.avg_pool2d(x, x.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        x = x * w

        x += self.shortcut(res)
        x = self.relu(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        self.fc1 = nn.Conv2d(planes, planes // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes // 16, planes, kernel_size=1)

    def forward(self, x):
        res = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        w = F.avg_pool2d(x, x.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        x = x * w

        x += self.shortcut(res)
        x = self.relu(x)

        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        self.inplanes = 64

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)

        self.layer1 = self._makelayer(block, 64, layers[0], stride=1)
        self.layer2 = self._makelayer(block, 128, layers[1], stride=2)
        self.layer3 = self._makelayer(block, 256, layers[2], stride=2)
        self.layer4 = self._makelayer(block, 512, layers[3], stride=2)

        self.avpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _makelayer(self, block, planes, blocks, stride):
        strides = [stride]+[1]*(blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes*block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


net = ResNet(BasicBlock, [3, 4, 6, 3])
if torch.cuda.is_available():
    print('use gpu')
    net = net.cuda()
optimizer = optim.SGD(net.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

trainSize = 12500
testSize = 2500
print('testSize:%d, trainSize:%d' % (testSize, trainSize))

print('Start to train...')
for epoch in range(100):

    train_loss = 0
    train_acc = 0
    test_loss = 0
    test_acc = 0
    # train
    for inputs, lables in train_loader:
        optimizer.zero_grad()
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            lables = lables.cuda()
        out = net(inputs)

        loss = loss_func(out, lables)
        train_loss += loss.item()

        pred = out.max(1, keepdim=True)[1]
        train_acc += pred.eq(lables.view_as(pred)).sum().item() / len(lables)

        loss.backward()
        optimizer.step()

    # test
    for inputs, lables in test_loader:
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            lables = lables.cuda()
        out = net(inputs)

        loss = loss_func(out, lables)
        test_loss += loss.item()

        pred = out.max(1, keepdim=True)[1]
        test_acc += pred.eq(lables.view_as(pred)).sum().item() / len(lables)

    print('[%d] [Train] train_loss:%.6f train_acc:%.6f   [Test] test_loss:%.6f test_acc:%.6f' % (epoch + 1,
                                                              train_loss / (trainSize),
                                                              train_acc / (trainSize),
                                                              test_loss / (testSize),
                                                              test_acc / (testSize)))




