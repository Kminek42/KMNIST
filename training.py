import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time

dev = torch.device("mps")
print(f"Using {dev} device.")

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))]
)

training_dataset = torchvision.datasets.KMNIST(
    root="./KMNIST",
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(
    dataset=training_dataset,
    batch_size=128,
    shuffle=True
)

# model
input_n = 28 * 28

hidden_n = 4096
output_n = 10
activation = nn.LeakyReLU()

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(input_n, hidden_n),
    activation,
    nn.Linear(hidden_n, hidden_n),
    activation,
    nn.Linear(hidden_n, output_n)
).to(dev)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

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

    learning_time = time.time() - t0
    remaining_time = learning_time / epoch * (epoch_n - epoch)
    print(f"Epoch: {epoch}, mean loss: {loss_sum / len(loader)}")
    print(f"Learning time: {time.time() - t0}, Time remaining: {remaining_time}\n")

torch.save(obj=model, f="model.pt")
print("Model saved.")
