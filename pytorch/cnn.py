# %%
import torch
import torchvision
from torchvision.transforms import transforms
import torch.nn.functional as F
import torch.optim as optim

from data_loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import model
torch.set_printoptions(linewidth=120)
from utils import *


# %%
data_loader = DataLoader('fashion_mnist',batch_size=10)

# %%
network = model.Network()


# %%
optimizer = optim.Adam(network.parameters(), lr = 1e-2)
# %%
# Traininig Loop

epoch_loss = 0;
epoch_correct_predictions = 0;
for batch in data_loader.train_loader:

    images, labels = batch # Get a batch

    preds = network(images) # Make predictions

    # Zero out the gradients for this batch
    optimizer.zero_grad();

    # Calulathe loss
    loss = F.cross_entropy(preds, labels);

    # Backward pass with autograd
    loss.backward()

    #update the weights
    optimizer.step()

    epoch_loss += loss.item();

    epoch_correct_predictions += get_num_correct(preds, labels)

print(f"Epoch 0, Loss = {epoch_loss} and number of correct prections were {epoch_correct_predictions}")




