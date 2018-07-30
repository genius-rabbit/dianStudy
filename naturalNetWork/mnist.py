'''
只有AMD显卡,没有使用GPU

'''
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as Data

train_data = torchvision.datasets.MNIST(
    '/home/geniusrabbit/PycharmProjects/TestPyTF/mnist', train=True, transform=transforms.ToTensor()
)

test_data = torchvision.datasets.MNIST(
    '/home/geniusrabbit/PycharmProjects/TestPyTF/mnist', train=False, transform=transforms.ToTensor()
)
print("train_data:", train_data.train_data.size())
print("train_labels:", train_data.train_labels.size())
print("test_data:", test_data.test_data.size())

train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=64)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 84)
        self.fc4 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

optimizer = optim.SGD(net.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()

for epoch in range(6):
    print('epoch{}'.format(epoch+1))

    train_loss = 0
    train_acc = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        batch_x, batch_y = Variable(batch_x), Variable(batch_y)

        out = net(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.data.item()

        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.data.item()

        loss.backward()
        optimizer.step()

    print('Train loss: {:.6f}, Acc: {:.6f}'.format(train_loss/(len(train_data)),
          train_acc/(len(train_data))))

    net.eval()
    eval_loss = 0
    eval_acc = 0
    for batch_x, batch_y in test_loader:

        batch_x, batch_y = Variable(batch_x), Variable(batch_y)

        out = net(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.data.item()

        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.data.item()

    print('Test loss:{:.6f}, Acc: {:.6f}'.format(eval_loss/(len(test_data)),
          eval_acc/(len(test_data))))