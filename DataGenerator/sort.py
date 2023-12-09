import torch
import torchvision
import json
import pickle


print("Loading data...")

# 读取数据
with open('./confidence.json', 'r') as f:
    data_order = json.load(f)

# 读取数据
data = torchvision.datasets.MNIST(
    root="../data", train=True,  transform=torchvision.transforms.ToTensor())

print("Sorting data...")

sorted_data_list = []

for currentData in data_order:
    # Convert currentData to a tensor
    sorted_data_list.append(data[currentData])


print("Saving data...")

pickle.dump(sorted_data_list, open("./data/sorted_MNIST_reversed.pkl", "wb"))

sorted_data_list.reverse()
pickle.dump(sorted_data_list, open("./data/sorted_MNIST.pkl", "wb"))
