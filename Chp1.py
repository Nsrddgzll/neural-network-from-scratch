import numpy as np
import pandas as pd
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]





weights = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]


weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]


# Perform a dot product using numpy when inputs are one dimensional
#outputs = np.dot(weights, inputs) + biases

# when we switch the position of inputs and weights, we need to transpose the weight matrix in order to perform dot product 
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer1_outputs)
print(layer2_outputs)



'''
layer_outputs = []

for neuron_weights, neuron_biases in zip(weights, biases):
    neuron_output = 0 # Output of given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_biases
    layer_outputs.append(neuron_output)

print(layer_outputs)
'''