from img2pixels import img_to_pixels
import math
import json


class Neuron:
    def __init__(self, value):
        self.value = value
        self.gradient = 0


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def calculate(img_path, weights, biases):
    input_layer = []
    for i in img_to_pixels(img_path):
        input_layer.append(Neuron(i))

    hidden_layer_one = [Neuron(0) for _ in range(81)]
    for i, neuron in enumerate(hidden_layer_one):
        neuron_sum = sum(input_layer[x].value * weights[0][i][x] for x in range(100))

        neuron.value = sigmoid(neuron_sum + biases[0][i])

    hidden_layer_two = [Neuron(0) for _ in range(81)]
    for i, neuron in enumerate(hidden_layer_two):
        neuron_sum = sum(hidden_layer_one[x].value * weights[1][i][x] for x in range(81))

        neuron.value = sigmoid(neuron_sum + biases[1][i])

    final_layer = [Neuron(0) for _ in range(10)]
    for i, neuron in enumerate(final_layer):
        neuron_sum = sum(hidden_layer_two[x].value * weights[2][i][x] for x in range(81))

        neuron.value = sigmoid(neuron_sum + biases[2][i])

    return [x.value for x in final_layer]


def load_weights_and_biases(filename):
    with open(filename, 'r') as f:
        weights, biases = json.load(f)
    return weights, biases


# HOW TO RUN THE MODEL 👇
# weights, biases = load_weights_and_biases("weights_biases.json")
# print(calculate("img.png", weights, biases))
