import torch
from torch import nn
import numpy as np
from typing import Final
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import helper_funcs as helper
import matplotlib.pyplot as plt

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

helper.createInputImage(X, y, name="spiral_input.png")

device = helper.determineDevice()