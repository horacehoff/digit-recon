import random
import json
from dataset import training_data, testing_data
from eval_accuracy import eval_accuracy
from img2pixels import img_to_pixels
from model import calculate, Neuron, sigmoid, sigmoid_derivative


def save_weights_and_biases(weights, biases, filename):
    with open(filename, 'w') as f:
        json.dump((weights, biases), f)


weights = [
    [[random.uniform(-1, 1) for _ in range(100)] for _ in range(81)],
    [[random.uniform(-1, 1) for _ in range(81)] for _ in range(81)],
    [[random.uniform(-1, 1) for _ in range(81)] for _ in range(10)]
]
biases = [
    [random.uniform(-1, 1) for _ in range(81)],
    [random.uniform(-1, 1) for _ in range(81)],
    [random.uniform(-1, 1) for _ in range(10)]
]


def calculate_cost(img_path, weights, biases, expected):
    output = calculate(img_path, weights, biases)
    cost = 0.0

    for x in range(10):
        cost += (output[x] - expected[x]) ** 2
    return cost


def backpropagate(img_path, weights, biases, expected, learning_rate):
    # SET->GET NEURON VALUES
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

    # CALCULATE OUTPUT LAYER ERROR GRADIENTS
    output_errors = [neuron.value - expected[i] for i, neuron in enumerate(final_layer)]
    for i, neuron in enumerate(final_layer):
        neuron.gradient = output_errors[i] * sigmoid_derivative(neuron.value)

    # CALCULATE LAYER #2 GRADIENTS
    for i, neuron in enumerate(hidden_layer_two):
        error = sum(final_layer[j].gradient * weights[2][j][i] for j in range(10))
        neuron.gradient = error * sigmoid_derivative(neuron.value)

    # CALCULATE LAYER #1 GRADIENTS
    for i, neuron in enumerate(hidden_layer_one):
        error = sum(hidden_layer_two[j].gradient * weights[1][j][i] for j in range(81))
        neuron.gradient = error * sigmoid_derivative(neuron.value)

    # UPDATE WEIGHTS/BIASES
    for i in range(10):
        for y in range(81):
            weights[2][i][y] -= learning_rate * final_layer[i].gradient * hidden_layer_two[y].value
        biases[2][i] -= learning_rate * final_layer[i].gradient

    for i in range(81):
        for y in range(81):
            weights[1][i][y] -= learning_rate * hidden_layer_two[i].gradient * hidden_layer_one[y].value
        biases[1][i] -= learning_rate * hidden_layer_two[i].gradient

    for i in range(81):
        for y in range(100):
            weights[0][i][y] -= learning_rate * hidden_layer_one[i].gradient * input_layer[y].value
        biases[0][i] -= learning_rate * hidden_layer_one[i].gradient


def train(training_data, weights, biases, learning_rate, epochs, stop_at_cost=0.0):
    for epoch in range(epochs):
        total_training_cost = 0
        total_testing_cost = 0
        for data in training_data:
            img_path, label = data
            expected = [0] * 10
            expected[label] = 1
            backpropagate(img_path, weights, biases, expected, learning_rate)
            total_training_cost += calculate_cost(img_path, weights, biases, expected)
        for data in testing_data:
            img_path, label = data
            expected = [0] * 10
            expected[label] = 1
            total_testing_cost += calculate_cost(img_path, weights, biases, expected)
        average_training_cost = total_training_cost / len(training_data)
        average_testing_cost = total_testing_cost / len(testing_data)
        print(f"Epoch {epoch + 1}/{epochs} -- Training Cost: {average_training_cost} -- Testing Cost: {average_testing_cost} -- Accuracy: {eval_accuracy(weights, biases)}%")
        if average_training_cost <= stop_at_cost:
            print("Cost limit reached - Saving weights...")
            save_weights_and_biases(weights, biases, "weights_biases.json")
            return
    print("Epochs limit reached - Saving weights...")
    save_weights_and_biases(weights, biases, "weights_biases.json")


epochs = 1000
learning_rate = 0.01
train(training_data, weights, biases, learning_rate, epochs)
