import torch
import torchvision
from model import AlexNet
from config import DefaultConfig

config = DefaultConfig()
net = AlexNet()
net.load_state_dict(torch.load("./model/basemodel/model.pth"))
net.eval()
net.to(config.device)


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])


def predict(image):
    with torch.no_grad():
        image = image.to(config.device)
        outputs = net(image)
        _, predicted = torch.max(outputs.data, 1)
        return (
            outputs[0],
            predicted
        )
