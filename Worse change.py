
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # CIFAR-10 images are 32x32
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 下载和加载训练集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# 下载和加载测试集
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

def train_model(model, trainloader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}')

# 初始化模型、损失函数和优化器
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 开始训练
train_model(model, trainloader, criterion, optimizer, num_epochs=5)

def add_noise(images, noise_level=0.5):
    noise = torch.randn(images.size()) * noise_level
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0, 1)

# 处理训练和测试集
noisy_trainset = [(add_noise(image), label) for image, label in trainset]
noisy_testset = [(add_noise(image), label) for image, label in testset]

# 创建新的 DataLoader
noisy_trainloader = DataLoader(noisy_trainset, batch_size=64, shuffle=True)
noisy_testloader = DataLoader(noisy_testset, batch_size=64, shuffle=False)

# 重新训练模型（可以选择使用添加噪声或遮挡后的数据集）
print("Training with noisy data:")
train_model(model, noisy_trainloader, criterion, optimizer, num_epochs=5)

# 验证模型性能
def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# 评估噪声数据集上的模型性能
noisy_accuracy = evaluate_model(model, noisy_testloader)
print(f'Accuracy on test set with noise: {noisy_accuracy:.2f}%')

