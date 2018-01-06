# AI for self Driving car

###
# Import the required libaries
###

import numpy as np
import random
# to save the brain
import os
# pytorch can handle dynamic graphs
import torch
# the nn package from torch contains all the tools for implementing neural networks
# well use this to get the three sensors inputs plus -orientation and +orientation
# and it will return q values and we will use softmax to return the 1 action to play
import torch.nn as nn
# contains the different function, typically the loss function, we will use
# uber loss from this module because it improves convergence
import torch.nn.functional as F
# the optimizer to perform SGD
import torch.optim as optim
# we import autograd just to take the Variable Class from it.
# we need it to do conversions from tensors to a variable that contains a gradient
import torch.autograd as autograd
from torch.autograd import Variable


###
# Create the architecture of the NN
###

class Network(nn.Module):
    # The init
    def __init__(self, input_size, nb_action):
        # this is just to be able to use all the tools from nn
        super(Network, self).__init__()

        # Variable declarations
        self.input_size = input_size
        self.nb_action = nb_action

        # The full connection variables
        #  - full connections means that all the input neurons from the input layer
        #    will be connected to all the neurons from the hidden layer
        #  - first arg is the number of features aka inputs, the second is the number of
        #    neurons in the hidden layer. It seems that 30 works ok
        # First connection is between input and hidden layer
        self.fc1 = nn.Linear(input_size, 30)
        # Second connection is between hidden layer and output
        self.fc2 = nn.Linear(30, nb_action)

    # The function that will do the forward propagation
    # and it will also return the Q values for the current state.
    def forward(self, state):
        # To get the activated neurons we will use a rectifier function
        # from the torch.nn.function
        activated_hidden_neurons = F.relu(self.fc1(state))
        # output neurons
        q_values = self.fc2(activated_hidden_neurons)
        return q_values


###
# Implementing Experience Replay
###

class ReplayMemory(object):
    # The init
    def __init__(self, capacity):

        # Variable declarations

        # How many events it will
        self.capacity = capacity
        # This will contain the last n events
        self.memory = []

    # It will append a new event in memory and will make sure that it does not exceed the capacity
    def push_events(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    # The sample function will take some random elements from the memory
    def sample(self, batch_size):
        # Basically because we have an input of the form
        # [[state1, action1, reward1], [state2, action2, reward2]]
        # but for our algorithm we need it to be of the form
        # [[state1, state2], [action1, action2], [reward1, reward2]
        # we'll use the zip() function which reshapes the array
        # the * operator here unpacks the array, sort of like ... (spread) in js
        sample = zip(*random.sample(self.memory, batch_size))

        # We will map over the samples and convert the torch tensor to a Variable containing
        # a tensor and gradient. We use torch.cat to make sure that we have it's first dimension
        # lined up (state action and reward) we use index 0 for that.
        return map(lambda x: Variable(torch.cat(x, 0)), sample)



