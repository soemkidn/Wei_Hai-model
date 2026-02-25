import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ===================== 2 使用pytorch库构建模型=====================
# 定义超参数
input_size = 784  # 28x28
hidden_size = 20
num_classes = 10
num_epochs = 1
batch_size = 64
learning_rate = 0.01

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义全连接神经网络
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 训练模型
def train(model, train_loader, optimizer, criterion, num_epochs):
    model.to(device)  # 将模型移动到GPU
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)  # 将数据移动到GPU
            # 前向传播
            outputs = model.forward(images.reshape(-1, 28 * 28))
            loss = criterion(outputs, labels)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')


# 测试模型
def predict(model, test_loader):
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据移动到GPU
            outputs = model.forward(images.reshape(-1, 28 * 28))
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

# 加载和预处理数据
train_dataset = datasets.MNIST(root='./', train=True, transform=transforms.ToTensor(), download=False)
test_dataset = datasets.MNIST(root='./', train=False, transform=transforms.ToTensor(), download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = NeuralNet(input_size, hidden_size, num_classes)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# 训练并测试模型
train(model, train_loader, optimizer, criterion, num_epochs=1)
predict(model, test_loader)