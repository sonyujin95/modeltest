import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import time


# ====================Dataset 준비=======================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(227),
    transforms.ToTensor(),  # 데이터 형태를 Tensor로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('PyTorch Version : ', torch.__version__, ' Device : ', device)

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, init_weights: bool = True):
        super(AlexNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, padding=0, stride=4),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),

            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.fclayer = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x:torch.Tensor):
        x = self.convnet(x)
        x = torch.flatten(x, 1)
        x = self.fclayer(x)
        return x

def init_weights(m):
    if type(m) not in [nn.ReLU, nn.LocalResponseNorm, nn.MaxPool2d, nn.Sequential, nn.Dropout, AlexNet]:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        m.bias.data.fill_(1)


alexnet = AlexNet(num_classes=1000)
alexnet.apply(init_weights)
alexnet = alexnet.to(device)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(alexnet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

start_time = time.time()
min_loss = int(1e9)
history = []
for epoch in range(100):
    epoch_loss = 0.0
    tk0 = tqdm(trainloader, total=len(trainloader), leave=False)
    for step, (inputs, labels) in enumerate(tk0, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = alexnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        history.append(loss.item())

    class_correct = list(0. for i in range(1000))
    class_total = list(0. for i in range(1000))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = alexnet(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(labels.size()[0]):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1

    tqdm.write('[Epoch : %d] train_loss: %.5f val_acc: %.2f' % (epoch + 1, epoch_loss / 157, sum(class_correct) / sum(class_total) * 100))
    if min_loss < epoch_loss:
        count+=1
        if count > 10:
            for g in optimizer.param_groups:
                g['lr']/=10
    else:
         min_loss = epoch_loss
         count = 0

print(time.time()-start_time)
print('Finished Training')
