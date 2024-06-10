import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Creates a matplotlib plot of the input moons and save the image
def createInputImage(x_values, y_values, name="input.png"):
    plt.figure(figsize=(10, 7))
    plt.scatter(x_values[:, 0], x_values[:, 1], c=y_values, cmap=plt.cm.inferno)
    plt.savefig(name)

# Create a matplotlib plot of the model's results of the training and test data points
def createOutputImage(model, x_train, y_train, x_test, y_test, name="output.png"):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plotDecisionBoundary(model, x_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plotDecisionBoundary(model, x_test, y_test)
    plt.savefig(name)

# Creates a white boundary line between group areas indentified by the model
def plotDecisionBoundary(model, x_values, y_values):

    model.to("cpu")
    x_values, y_values = x_values.to("cpu"), y_values.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = x_values[:, 0].min() - 0.1, x_values[:, 0].max() + 0.1
    y_min, y_max = x_values[:, 1].min() - 0.1, x_values[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y_values)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.inferno, alpha=0.7)
    plt.scatter(x_values[:, 0], x_values[:, 1], c=y_values, s=40, cmap=plt.cm.inferno)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# Calculates the accuracy of the model
def accuracy_fn(input_values, predicted_values):
    correctGuesses = torch.eq(input_values, predicted_values).sum().item()
    return ((correctGuesses / len(predicted_values)) * 100)

# Determines if pytorch can access GPU for 
def determineDevice():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
    
def printTrainTime(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time

if __name__ == "__main__":
    print("Run one of the learn python files to see the ML models in action")