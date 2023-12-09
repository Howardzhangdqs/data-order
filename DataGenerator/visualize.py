import matplotlib.pyplot as plt
import torchvision
import json

print("Loading data...")

data = torchvision.datasets.MNIST(
    root="../data", train=True, transform=torchvision.transforms.ToTensor())

print("Loading confidence...")

with open("./confidence.json", "r") as f:
    data_order = json.load(f)

print("Plotting...")

# 展示置信度最高和最低20的数据
plt.figure(figsize=(10, 10))
for i in range(20):
    plt.subplot(4, 10, i+1)
    plt.title(f"{data[data_order[i]][1]}")
    plt.imshow(data[data_order[i]][0][0].cpu().numpy(), cmap="gray")
    plt.axis("off")

    plt.subplot(4, 10, i+21)
    plt.title(f"{data[data_order[-i-1]][1]}")
    plt.imshow(data[data_order[-i-1]][0][0].cpu().numpy(), cmap="gray")
    plt.axis("off")

plt.show()
