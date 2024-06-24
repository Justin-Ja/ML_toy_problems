import torch
import helperFuncs as helper
import argparse
from torch import nn
from typing import Final
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer

parser = argparse.ArgumentParser(prog="\nA ML program that trains to learn a binary classification problem\n")

parser.add_argument('-N', '--samples', help='Number of samples to generate', 
                    default=1000, type=int)
parser.add_argument('-n', '--noise', help='A value between 0 and 1 of how spread out the points are. A higher value means points are further apart.', 
                    default=0.08, type=float)
parser.add_argument('-e', '--epochs', help='The number of loops should the model train/test for', 
                    default=250, type=int)
parser.add_argument('-l', '--learning_rate', help='The rate that the optimizer will update the model\'s parameters at.', 
                    default=0.1, type=float)
parser.add_argument('-u', '--units', help='Number of neurons to use per ML layer. Higher value helps the model be more accurate, at the cost of training time\n', 
                    default=16, type=int)
parser.add_argument('-s', '--seed', help='Value of seed for RNG', 
                    default=42, type=int)

args = parser.parse_args()

# Hyperparameters, adjust these to effect the generated input and the effeciency of the model
N_SAMPLES: Final[int] = args.samples
NOISE: Final[int] = args.noise
RAND_SEED: Final[int] = args.seed
HIDDEN_UNITS: Final[int] = args.units
LR: Final[float] = args.learning_rate
EPOCHS: Final[int] = args.epochs

# Creating and adjusting data into suitable training/testing tensors
points, groups = make_moons(N_SAMPLES, noise=NOISE, random_state=RAND_SEED)

points = torch.from_numpy(points).type(torch.float)
groups = torch.from_numpy(groups).type(torch.float)

points_train, points_test, groups_train, groups_test = train_test_split(points, groups, test_size=0.2)

# Allows for device agnostic code
device = helper.determineDevice()

helper.createInputImage(points, groups, name="moon_input.png")

# Consider possibly move this to its own file, and jsut have a new file for each class of model?
class MoonModel(nn.Module):
    def __init__(self, input_features: int, output_features: int, hidden_units: int = 5):
        super().__init__()

        # Model has three hidden layers, with default of 5 neurons (hidden_units)
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_features),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)
    
model_0 = MoonModel(input_features = 2, output_features = 1, hidden_units = HIDDEN_UNITS)

#Binary groups so we can use BCE for loss calculation
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_0.parameters(), LR) 


# Setup before training: Move data to same device, set seed and timer, and output printing setup
points_train, groups_train = points_train.to(device), groups_train.to(device)
points_test, groups_test = points_test.to(device), groups_test.to(device)
torch.manual_seed(RAND_SEED)
torch.cuda.manual_seed(RAND_SEED)
startTime = timer()

modulusPrintValue = int(EPOCHS / 20)

for epoch in range(EPOCHS + 1): #There's a plus one to get the final epoch stats printed out later
    ### TRAINING
    model_0.train()

    group_logits = model_0(points_train).squeeze()
    group_pred = torch.round(torch.sigmoid(group_logits)) # logits -> prediction probabilities -> prediction labels

    loss = loss_fn(group_logits, groups_train)
    accuracy = helper.accuracy_fn(groups_train, group_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ### TESTING
    model_0.eval()
    with torch.inference_mode():
        # Foward pass
        test_logits = model_0(points_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
        
        # Calcuate loss and accuracy
        test_loss = loss_fn(test_logits, groups_test)
        test_accuracy = helper.accuracy_fn(groups_test, test_pred)

    if epoch % modulusPrintValue == 0:
        # Print out current status of the model
        print(f"Epoch: {epoch} | Accuracy: {accuracy: 0.2f}% | Loss: {loss: 0.5f} | - | Test Accuracy: {test_accuracy: 0.2f}% | Test Loss: {test_loss: 0.5f}")

# One final print to see the final state of the model after training
print(f"Epoch: {epoch} | Accuracy: {accuracy: 0.2f}% | Loss: {loss: 0.5f} | - | Test Accuracy: {test_accuracy: 0.2f}% | Test Loss: {test_loss: 0.5f}")

endTime = timer()
helper.printTrainTime(startTime, endTime, device)
helper.createOutputImage(model_0, points_train, groups_train, points_test, groups_test, name="moon_output.png")
print("Created input and output images of the moon data successfully!")
