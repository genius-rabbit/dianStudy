import torch
import math
import torchvision
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data.dataloader as Data
import tensorflow as tf

useTensorboard = False

logdir = '/output/'
testStep = 500
netSize = [2, 2, 2, 2]
epochSize = 100
batch_size = 128
root = './cifar'
# root = '/home/geniusrabbit/PycharmProjects/TestPyTF/cifar100'


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


def ResNet18():
    return ResNet(BasicBlock, netSize)


def train(model, data, target, loss_func, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    predictions = output.max(1, keepdim=True)[1]
    correct = predictions.eq(target.view_as(predictions)).sum().item()
    acc = correct / len(target)
    loss = loss_func(output, target)
    loss.backward()
    optimizer.step()
    return acc, loss


def test(model, test_loader, loss_func, use_cuda):
    model.eval()
    acc_all = 0
    loss_all = 0
    step = 0
    with torch.no_grad():
        for data, target in test_loader:
            step += 1
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            predictions = output.max(1, keepdim=True)[1]
            correct = predictions.eq(target.view_as(predictions)).sum().item()
            acc = correct / len(target)
            loss = loss_func(output, target)
            acc_all += acc
            loss_all += loss
    return acc_all / step, loss_all / step


def main():
    # tensorboard
    if useTensorboard:
        # use tensorboard

    use_cuda = torch.cuda.is_available()
    train_loader = Data.DataLoader(
        torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=
        transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = Data.DataLoader(
        torchvision.datasets.CIFAR100(root=root, train=False,transform=
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
        batch_size=batch_size
    )

    # define network
    model = ResNet18()
    if use_cuda:
        model = model.cuda()
    print(model)
    # define loss function
    ce_loss = torch.nn.CrossEntropyLoss()

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    # writer = SummaryWriter()
    # start train
    train_step = 0
    for _ in range(epochSize):
        for data, target in train_loader:
            train_step += 1
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            acc, loss = train(model, data, target, ce_loss, optimizer)
            if train_step % 100 == 0:
                print('[Train]: Step: {}, Loss: {:.6f}, Accuracy: {:.6f}'.format(train_step, loss, acc))
                #writer.add_scalars('data/group', {'train_loss':loss, 'train_acc':acc}, train_step/100)
            if train_step % testStep == 0:
                acc, loss = test(model, test_loader, ce_loss, use_cuda)
                print('[Test ]set: Step: {}, Loss: {:.6f}, Accuracy: {:.6f}\n'.format(train_step, loss, acc))
                #writer.add_scalars('data/group', {'test_loss':loss, 'test_acc':acc}, train_step/100)

    #writer.export_scalars_to_json("./all_scalars.json")
    #writer.close()

if __name__ == '__main__':
    main()



