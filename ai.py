# AI for self Driving car

# Import the required libaries

import numpy as np
import random
# to save the brain
import os
# pytorch can handle dynamic graphs
import torch
# the nn package from torch contains all the tools for implementing neural networks
# well use this to get the three sensors inputs plus -orientation and +orientation
# and it will return q values and we will use softmax to return the 1 action to play
import torch.nn
# contains the different function, typically the loss function, we will use
# uber loss from this module because it improves convergence
import torch.nn.functional as f
# the optimizer to perform SGD
import torch.optim as optim
# we import autograd just to take the Variable Class from it.
# we need it to do conversions from tensors to a variable that contains a gradient
import torch.autograd as autograd
from torch.autograd import Variable

