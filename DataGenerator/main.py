import torchvision
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import operator
import json
from eval import predict, transform


# 将mnist数据集转换为图片列表
data = torchvision.datasets.MNIST(
    root="../data", train=True, transform=transform)

dataLoader = torch.utils.data.DataLoader(
    dataset=data, batch_size=1, num_workers=0)


data_confidence = []


def confidence(res, label):
    return res[label] * 2 - res.sum()


# 保存图片
for i, (image, label) in enumerate(dataLoader):
    _, res = predict(image)
    # print(_.sum(), res, res.to("cuda:0") == label.to("cuda:0"))

    data_confidence.append(confidence(_, label))

    # print(_, confidence(_, label))

    # plt.imshow(image[0][0].cpu().numpy(), cmap="gray")
    # plt.show()

    if (i > 1000):
        break

    if (i % 1000 == 0):
        print(i)


# 对数据进行排序
data_order = sorted(list(enumerate(data_confidence)),
                    key=operator.itemgetter(1))

print(len(data_order))

# 保存数据
with open("./confidence.json", "w") as f:
    json.dump(list(map(lambda x: x[0], data_order)), f)
