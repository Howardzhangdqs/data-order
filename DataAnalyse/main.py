"""
将实验数据用折线图表示出来

数据文件名分为以下三类：
acc_2023-12-08-16-56-42.json
acc_2023-12-08-17-02-47_shuffled.json
acc_2023-12-08-17-22-11_reversed.json
都是"acc_"开头，以".json"结尾，中间的时间为实验结束时间，后面的"_shuffled"和"_reversed"分别表示是否打乱数据集和是否反转数据集

数据文件格式为：
{
    "acc": [ACC列表],
    "loss": [Loss列表]
}

绘制折线图时，横坐标为迭代次数，纵坐标为ACC或Loss，ACC与Loss分别用两张图表示，左边表示ACC，右边表示Loss，没有打乱数据集的实验用蓝色表示，打乱数据集的实验用红色表示，反转数据集的实验用绿色表示
"""
import matplotlib.pyplot as plt
import os
import json

res_file = os.listdir("./result/")

res_file = [i for i in res_file if i.endswith(".json")]

colors = {
    "": "#00f7",  # 蓝色
    "shuffled.json": "#f007",  # 红色
    "reversed.json": "#0f07",  # 绿色
}

fig = plt.figure(figsize=(10, 5))
fig.suptitle("ACC and Loss")

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

for file in res_file:
    with open(f"./result/{file}", "r") as f:
        data = json.load(f)
    acc = data["acc"]
    loss = data["loss"]
    color = colors[""
                   if not file.split("_")[-1] in colors else file.split("_")[-1]]

    # print(color, file.split("_")[-1])

    ax1.plot(range(len(acc)), acc, color=color)
    ax2.plot(range(len(loss)), loss, color=color)

ax1.set_xlabel("Iteration")
ax1.set_ylabel("ACC")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Loss")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.show()
