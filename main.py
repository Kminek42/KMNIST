import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import learining_time_est as lte
import sys


def set_device(device_name):
    if device_name == "cpu":
        print("Using cpu")
        return torch.device("cpu")

    if device_name == "cuda":
        if torch.cuda.is_available():
            print("Using cuda")
            return torch.device("cuda")
        
        else:
            print("Cuda not avaiable, using cpu instead")
            return torch.device("cpu")


    elif device_name == "mps":
        if torch.backends.mps.is_available():
            print("Using mps")
            return torch.device("mps")
        
        else:
            print("MPS not avaiable, using cpu instead")
            return torch.device("cpu")

    else:
        print("Invalid device name, using cpu instead")
        return torch.device("cpu")

if len(sys.argv) != 3:
    print("Usage: main.py [train / test] [cpu / cuda / mps]")
    exit()

dev = set_device(sys.argv[2])

if sys.argv[1] == "train":
    loader = DataLoader(
        dataset=torchvision.datasets.KMNIST(
            root="./KMNIST",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor()
        ),
        batch_size=256,
        shuffle=True
    )
    activation = nn.LeakyReLU()
    model = nn.Sequential(
    # 28 * 28

        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), padding="same"),  # 28 * 28 * 16
        activation,
        nn.AvgPool2d(kernel_size=(2, 2), stride=2),  # 14 * 14 * 16 

        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), padding="same"),  # 14 * 14 * 32
        activation,
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), padding="same"),  # 14 * 14 * 32
        activation,
        nn.AvgPool2d(kernel_size=(2, 2), stride=2),  # 7 * 7 * 32 

        nn.Flatten(),
        nn.Linear(7 * 7 * 32, 512),
        activation,
        nn.Linear(512, 512),
        activation,
        nn.Linear(512, 10)
    ).to(dev)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)

    epoch_n = 10

    t0 = time.time()
    for epoch in range(1, epoch_n + 1):
        loss_sum = 0
        for inputs, targets in iter(loader):
            inputs, targets = inputs.to(dev), targets.to(dev)
            outputs = model.forward(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            loss_sum += loss

        print(f"\nEpoch: {epoch}, mean loss: {loss_sum / len(loader)}")
        lte.show_time(start_timestamp=t0, progres=epoch/epoch_n)

    torch.save(obj=model.to("cpu"), f="model.pt")
    print("Model saved.")

elif sys.argv[1] == "test":
    model = torch.load("./model.pt").to(dev)
    loader = DataLoader(
        dataset=torchvision.datasets.KMNIST(
            root="./KMNIST",
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor()
        ),
        batch_size=256,
        shuffle=True
    )
    good = 0
    all = 0
    for inputs, targets in iter(loader):
        inputs, targets = inputs.to(dev), targets.to(dev) 
        outputs = model.forward(inputs)
        for i in range(len(outputs)):
            if torch.argmax(outputs[i]) == targets[i]:
                good += 1
            all += 1
        
        print(all/10000)
        

    print(f"Accuracy: {good / all}")

else:
    print("Usage: main.py [train / test] [cpu / cuda / mps]")
