# Import Libraries

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt
# Generate the Dataset
def generate_data(n_samples=2000):
  X = torch.zeros(n_samples, 2)
  y = torch.zeros(n_samples, dtype=torch.long)

  # Generate samples from two Gaussian distributions
  X[:n_samples//2] = torch.randn(n_samples//2, 2) + torch.Tensor([3,2])
  X[n_samples//2:] = torch.randn(n_samples//2, 2) + torch.Tensor([-3,2])

  # Labels
  for i in range(X.shape[0]):
    if X[i].norm() > math.sqrt(13):
      y[i] = 1

  X[:, 1] = X[:, 1] - 2

  return X, y

data, labels = generate_data()


class Expert(nn.Module):
  # TODO: Implement the Expert class
    def __init__(self):
        super(Expert, self).__init__()
        self.linear = nn.Linear(2, 1)
    def forward(self, x):
        return self.linear(x)
class GatingNetwork(nn.Module):
    def __init__(self, num_experts):
        super(GatingNetwork, self).__init__()
        self.linear = nn.Linear(2, num_experts)
    def forward(self, x):
        return F.softmax(self.linear(x), dim=1)
class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts =2 ):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList([Expert() for i in range(num_experts)])
        self.gating_network = GatingNetwork(num_experts)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        g = self.gating_network(x)
        y_hat = torch.zeros(x.shape[0], 1)
        for i in range(len(self.experts)):
            y_hat += g[:, i].view(-1, 1) * self.experts[i](x)
        return self.sigmoid(y_hat)
# Define the model, loss, and optimizer
model = MixtureOfExperts()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

# Convert data and labels to float tensors
data_tensor = data.float()
labels_tensor = labels.view(-1, 1).float()

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass

    y_hat = model(data_tensor)
    
    loss = criterion(y_hat, labels_tensor)
    acc = ((y_hat > 0.5) == labels_tensor).sum().float() / len(labels_tensor)
    #acc = (labels== predicted).sum().float() / len(labels)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Decay the learning rate
    scheduler.step()

    # Print out the loss and accuracy
    print("[EPOCH]: %i, [LOSS]: %.6f, [ACCURACY]: %.3f" % (epoch, loss.item(), acc))

