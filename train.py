import torch
import torchvision
from torchvision.transforms import transforms
import torch.nn.functional as F
import torch.optim as optim
import model
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


###################### Get the network #############################
network = model.Network()

##################### Initialize the optimizer #####################
optimizer = optim.Adam(network.parameters(), lr = 1e-2)

###################### Training Loop ###############################
epoch_loss = 0;
epoch_correct_predictions = 0;

for epoch in range(2):
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

    print(f"Epoch {epoch}, Loss = {epoch_loss} and number of correct prections were {epoch_correct_predictions}")



