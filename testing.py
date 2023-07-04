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
    torchvision.transforms.Normalize((0.0,), (1.0,))]
)

testing_dataset = torchvision.datasets.KMNIST(
    root="./KMNIST",
    train=False,
    download=True,
    transform=transform
)

loader = DataLoader(
    dataset=testing_dataset,
    batch_size=256,
    shuffle=True
)

model = torch.load("./model.pt").to("cpu")

good = 0
all = 0
for inputs, targets in iter(loader):
    # inputs, targets = inputs.to(dev), targets.to(dev) 
    outputs = model.forward(inputs)
    for i in range(len(outputs)):
        if torch.argmax(outputs[i]) == targets[i]:
            good += 1
        all += 1
    
    print(all)
    

print(f"Accuracy: {good / all}")
