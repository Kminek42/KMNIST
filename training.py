import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import learining_time_est as lte

dev = torch.device("mps")
print(f"Using {dev} device.")

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.0,), (1.0,))]
)

training_dataset = torchvision.datasets.KMNIST(
    root="./KMNIST",
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(
    dataset=training_dataset,
    batch_size=256,
    shuffle=True
)

# model
activation = nn.LeakyReLU()

model = nn.Sequential(
    # 28 * 28

    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), padding="same"),  # 28 * 28 * 16
    activation,
    nn.AvgPool2d(kernel_size=(2, 2), stride=2),  # 14 * 14 * 16 

    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), padding="same"),
    activation,
    nn.AvgPool2d(kernel_size=(2, 2), stride=2),  # 7 * 7 * 16 

    nn.Flatten(),
    nn.Linear(7 * 7 * 16, 84),
    activation,
    nn.Linear(84, 10)
).to(dev)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)

epoch_n = 20

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

torch.save(obj=model, f="model.pt")
print("Model saved.")
