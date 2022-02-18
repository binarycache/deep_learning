import torch
import torchvision
from torchvision.transforms import transforms
import torch.nn.functional as F
import torch.optim as optim
from import model
import utils
from data_loader import DataLoader

torch.set_grad_enabled(True)

## The Training Process

# 1. Get batch from the training set
# 2. Pass this batch to the network
# 3. Calcuate the loss
# 4. Calculate gradient of loss functions w.r.t. the weights.
# 5. Update the weights using gradient descent
# 6. Repeat steps 1-5 for a single epoch.
# 7. Repeat 4-6

###################### Get the data first ###########################
data_loader = DataLoader('fashion_mnist',batch_size=64)




