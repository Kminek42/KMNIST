import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time

dev = torch.device("cpu")
print(f"Using {dev} device.")

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))]
)

testing_dataset = torchvision.datasets.MNIST(
    root="./MNIST",
    train=False,
    download=True,
    transform=transform
)
model = torch.load("./model.pt").to("cpu")

good = 0
all = 0
for inputs, targets in testing_dataset:
    # inputs, targets = inputs.to(dev), targets.to(dev)
    outputs = model.forward(inputs)
    if torch.argmax(outputs) == targets:
        good += 1
    
    all += 1

print(f"Accuracy: {good / all}")
