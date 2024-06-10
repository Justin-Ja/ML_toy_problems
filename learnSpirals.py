import torch
from torch import nn
import numpy as np
from typing import Final
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import helperFuncs as helper

RAND_SEED: Final[int] = 50
LR: Final[float] = 0.01  #If using 0.1, turn epochs down to ~10,000 otherwise accuracy starts decreasing / loss increases
EPOCHS: Final[int] = 25000

#Code to generate sprials from here: https://cs231n.github.io/neural-networks-case-study/

N = 100 # number of points per class
D = 2 # dimensionality
K = 6 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,2,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.175 # theta - how spread out the dots are from their "central line"
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

input_X = torch.from_numpy(X).type(torch.float)
input_y = torch.from_numpy(y).type(torch.float)

x_train, x_test, y_train, y_test = train_test_split(input_X, input_y, test_size=0.2, random_state=RAND_SEED)

helper.createInputImage(X, y, name="spiral_input.png")

device = helper.determineDevice()

class SpiralModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(), 
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )

    def forward(self, x):
       return self.linear_layer_stack(x)

#Input is x,y coords so 2, and output is number of classes, which is 6
spiral_model_0 = SpiralModel(input_features = 2, output_features = 6, hidden_units=32).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(spiral_model_0.parameters(), LR)

# Ensure all data is on the same device
x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)

# We use CrossEntropyLoss for the loss function, so we need to use longs instead of floats
y_train = y_train.to(torch.long)
y_test = y_test.to(torch.long)

torch.manual_seed(RAND_SEED)

for epoch in range(EPOCHS + 1):
    ### TRAINING
    spiral_model_0.train()

    y_logits = spiral_model_0(x_train) # model outputs raw logits 
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) # go from logits -> prediction probabilities -> prediction labels

    loss = loss_fn(y_logits, y_train) 
    accuracy = helper.accuracy_fn(y_train, y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ### TESTING
    spiral_model_0.eval()
    with torch.inference_mode():    
        test_logits = spiral_model_0(x_test) 
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, y_test)
        test_accuracy = helper.accuracy_fn(y_test, test_pred)

    if epoch % 2500 == 0:
        # Print out current status of the model
        print(f"Epoch: {epoch} | Accuracy: {accuracy: 0.2f}% | Loss: {test_loss: 0.5f} | - | Test Accuracy: {test_accuracy: 0.2f}% | Test Loss: {test_loss: 0.5f}")

helper.createOutputImage(spiral_model_0, x_train, y_train, x_test, y_test, name="spiral_output.png" )