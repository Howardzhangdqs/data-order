import torch
import torchvision
from model import AlexNet
from config import DefaultConfig

config = DefaultConfig()
net = AlexNet()
net.to(config.device)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

# MNIST数据加载
trainData = torchvision.datasets.MNIST(
    root=config.dataPath, train=True, download=True, transform=transform)
testData = torchvision.datasets.MNIST(
    root=config.dataPath, train=False, download=True, transform=transform)

trainLoader = torch.utils.data.DataLoader(
    dataset=trainData, batch_size=config.batchSize, shuffle=True, num_workers=config.numWorkers)
testLoader = torch.utils.data.DataLoader(
    dataset=testData, batch_size=config.batchSize, shuffle=True, num_workers=config.numWorkers)

# 损失函数和优化器
lossFunc = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)


# 测试
def ACC(net):
    correct = 0
    total = 0
    for images, labels in testLoader:
        images = images.to(config.device)
        labels = labels.to(config.device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print("Accuracy of the network on the %d test images: %.4f%%" %
          (total, (100 * correct / total)))
    return (100 * correct / total)


acc = 0

# 训练
for epoch in range(config.epoch):
    for i, (images, labels) in enumerate(trainLoader):
        images = images.to(config.device)
        labels = labels.to(config.device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = lossFunc(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" %
                  (epoch+1, config.epoch, i + 1, len(trainData) // config.batchSize, loss.item()))

    # 如果ACC增大，则保存模型
    currentACC = ACC(net)
    if currentACC > acc:
        torch.save(net.state_dict(), config.modelPath)
        acc = currentACC
        print("Model has been saved.")
