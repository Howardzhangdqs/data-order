import torch
import torchvision
import matplotlib.pyplot as plt
import pickle
import os
import sys
import datetime
import json
import time
from rich import print
from config import DefaultConfig


try:
    print(os.getcwd(), os.path.join(os.getcwd(), "models"))
    sys.path.append(os.path.join(os.getcwd(), "models"))
    from AlexNet import Net
except:
    print("Please run this script in the root directory.")
    exit(1)

config = DefaultConfig()

shuffle = False
reverse = True


testData = torchvision.datasets.MNIST(
    root=config.dataPath, train=False, download=True, transform=torchvision.transforms.ToTensor())

trainData = pickle.load(
    open("./data/sorted_MNIST" + ("_reversed" if reverse else "") + ".pkl", "rb"))


trainLoader = torch.utils.data.DataLoader(
    dataset=trainData, batch_size=config.batchSize, shuffle=shuffle, num_workers=config.numWorkers)
testLoader = torch.utils.data.DataLoader(
    dataset=testData, batch_size=4096, shuffle=shuffle, num_workers=config.numWorkers)


def train():

    net = Net()
    net.to(config.device)

    # 损失函数和优化器
    lossFunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)

    # 测试

    def ACC(net):

        net.eval()

        with torch.no_grad():
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

        net.train()

        return (100 * correct / total)

    acc = 0

    acc_list = []
    all_loss_list = []
    loss_list = []

    for epoch in range(config.epoch):
        for i, (images, labels) in enumerate(trainLoader):
            images = images.to(config.device)
            labels = labels.to(config.device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = lossFunc(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" %
                      (epoch + 1, config.epoch, i + 1, len(trainData) // config.batchSize, loss.item()))
                loss_list.append(loss.item())
                acc = float(ACC(net))
                acc_list.append(acc)

            all_loss_list.append(loss.item())

    json.dump({
        "acc": acc_list,
        "loss": all_loss_list
    }, open(
        "./result/acc_" +
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") +
        ("_shuffled" if shuffle else "") +
        ("_reversed" if reverse else "") + ".json", "w"))

    # 训练时使用plt双Y轴折线图实时显示ACC与Loss

    # plt.figure(figsize=(10, 5))
    # plt.title("ACC and Loss")

    # ax1 = plt.gca()
    # ax2 = ax1.twinx()

    # ax1.plot(acc_list, color="red", label="ACC")
    # ax2.plot(loss_list, color="blue", label="Loss")

    # ax1.set_xlabel("Iteration")
    # ax1.set_ylabel("ACC")

    # ax2.set_ylabel("Loss")

    # ax1.legend(loc="upper left")
    # ax2.legend(loc="upper right")

    # plt.show()

    time.sleep(5)


for i in range(0, 10):
    train()
