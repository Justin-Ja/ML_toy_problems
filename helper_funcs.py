import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def createMoonInputImage(x_values, y_values, name="moon_input.png"):
    plt.figure(figsize=(10, 7))
    plt.scatter(x_values[:, 0], x_values[:, 1], c=y_values, cmap=plt.cm.jet)
    plt.savefig(name)