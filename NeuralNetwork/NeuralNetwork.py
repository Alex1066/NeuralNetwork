import numpy as np
from functools import lru_cache
from random import randint
xor = [
    {
        'input': [0, 0],
        'target': [0]
    },
    {
        'input': [0, 1],
        'target': [1]
    },
    {
        'input': [1, 0],
        'target': [1]
    },
    {
        'input': [1, 1],
        'target': [0]
    }
]


class NeuralNetwork:
    def __init__(self, nodes_structure, learning_rate):
        self.nodes_structure = nodes_structure
        self.weights = []
        self.biases = []
        for i in range(len(self.nodes_structure) - 1):
            self.weights.append(np.random.rand(self.nodes_structure[i + 1], self.nodes_structure[i]))
            self.biases.append(np.random.rand(self.nodes_structure[i + 1], 1))
        self.weights[0][0][0] = 0.59928915
        self.weights[0][0][1] = 0.49381988
        self.weights[0][1][0] = 0.00993668
        self.weights[0][1][1] = 0.99211803
        self.weights[1][0][0] = 0.05656974
        self.weights[1][0][1] = 0.64863012
        self.biases[0][0] = 0.41595131
        self.biases[0][1] = 0.45151688
        self.biases[1][0] = 0.37831008

        self.learning_rate = learning_rate
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.sigmoid_derivative = lambda y: self.sigmoid(y) * (1 - self.sigmoid(y))
        self.outputs = []
        # The first layer does not have an weighted sum so this will be
        # one index lower than the rest of the matrices.
        self.weighted_sums = []
        self.nodes_values = []
        self.targets = np.array([])
        self.correct_weight_deltas = []
        self.correct_bias_deltas = []
        for i in range(len(self.nodes_structure) - 1):
            self.correct_weight_deltas.append(np.zeros((self.nodes_structure[i + 1], self.nodes_structure[i])))
            self.correct_bias_deltas.append(np.zeros((self.nodes_structure[i + 1], 1)))

    def feedforward(self, inputs):
        layer_values = np.array(inputs)
        layer_values.shape = (self.nodes_structure[0], 1)
        nodes_values = []
        weighted_sums = []
        nodes_values.append(layer_values)
        for i in range(len(self.nodes_structure) - 1):
            layer_values = self.weights[i] @ layer_values + self.biases[i]
            weighted_sums.append(layer_values)
            layer_values = self.sigmoid(layer_values)
            nodes_values.append(layer_values)
        self.nodes_values = np.array(nodes_values)
        self.weighted_sums = np.array(weighted_sums)

        return layer_values

    # @lru_cache(maxsize=1000)
    def derivative_of_cost_with_respect_of_node(self, layer, index):
        if layer == len(self.nodes_structure) - 1:
            value = 2 * (self.outputs[index][0] - self.targets[index][0])
            return value
        else:
            # For every node in the next layer.
            # print('value for deriv of node ', layer, ' with the index ', index, ' is: ')
            for h in range(self.nodes_structure[layer + 1]):
                return self.weights[layer][h][index] * self.sigmoid_derivative(self.weighted_sums[layer][h]) \
                        * self.derivative_of_cost_with_respect_of_node(layer + 1, h)

    def derivative_of_cost_with_respect_of_weight(self, layer, to_node, from_node):
        index1 = to_node
        index2 = from_node
        result = self.nodes_values[layer][index2][0] * self.sigmoid_derivative(self.weighted_sums[layer][index1][0]) \
                 * self.derivative_of_cost_with_respect_of_node(layer + 1, index1)
        return result

    def derivative_of_cost_with_respect_of_bias(self, layer, index):
        result = self.sigmoid_derivative(self.weighted_sums[layer][index]) \
                 * self.derivative_of_cost_with_respect_of_node(layer + 1, index)
        return result

    def train(self, inputs, targets):
        # Transform the given array into a column ndarray.
        self.targets = np.array(targets)
        self.targets.shape = (len(targets), 1)
        self.outputs = self.feedforward(inputs)
        layer_deltas = []
        layer_bias_deltas = []
        for i in range((len(self.nodes_structure) - 2), -1, -1):
            layer_deltas.append(np.zeros((self.nodes_structure[i + 1], self.nodes_structure[i])))
            layer_bias_deltas.append(np.zeros((self.nodes_structure[i + 1], 1)))
            for j in range(self.nodes_structure[i + 1]):
                for k in range(self.nodes_structure[i]):
                    layer_deltas[len(self.nodes_structure) - i - 2][j][k] = self.learning_rate * self.derivative_of_cost_with_respect_of_weight(i, j, k)
                layer_bias_deltas[len(self.nodes_structure) - i - 2][j] = self.learning_rate * self.derivative_of_cost_with_respect_of_bias(i, j)
        correct_weight_deltas = []
        correct_bias_deltas = []
        for i in range(len(layer_deltas)):
            correct_weight_deltas.append(layer_deltas[len(layer_deltas) - 1 - i])
            correct_bias_deltas.append(layer_bias_deltas[len(layer_deltas) - 1 - i])
        for i in range(len(self.nodes_structure) - 1):
            for j in range(self.nodes_structure[i + 1]):
                for k in range(self.nodes_structure[i]):
                    self.correct_weight_deltas[i][j][k] += correct_weight_deltas[i][j][k]
                self.correct_bias_deltas[i][j] += correct_bias_deltas[i][j]

    def change_weights_and_bias(self):
        for i in range(len(self.nodes_structure) - 1):
            for j in range(self.nodes_structure[i + 1]):
                for k in range(self.nodes_structure[i]):
                    to_change_weight = self.correct_weight_deltas[i][j][k]/100
                    self.weights[i][j][k] -= to_change_weight
                to_change_bias = self.correct_bias_deltas[i][j]/100
                self.biases[i][j] -= to_change_bias
        self.correct_weight_deltas = []
        self.correct_bias_deltas = []

        for i in range(len(self.nodes_structure) - 1):
            self.correct_weight_deltas.append(np.zeros((self.nodes_structure[i + 1], self.nodes_structure[i])))
            self.correct_bias_deltas.append(np.zeros((self.nodes_structure[i + 1], 1)))


nn = NeuralNetwork([2, 2, 1], 0.3)
for i in range(10000):
    if i % 100 == 0:
        print(i)
    for j in range(100):
        pos = randint(0, 3)
        nn.train(xor[pos]['input'], xor[pos]['target'])
    nn.change_weights_and_bias()

print(nn.feedforward(np.array([0, 0])))
print(nn.feedforward(np.array([1, 1])))
print(nn.feedforward(np.array([1, 0])))
print(nn.feedforward(np.array([0, 1])))
