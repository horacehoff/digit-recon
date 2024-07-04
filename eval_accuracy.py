from dataset import testing_data
from model import load_weights_and_biases, calculate


def eval_accuracy(weights, biases):
    order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    losses = []
    for img in testing_data:
        output = calculate(img[0], weights, biases)
        if order[output.index(max(output))] != img[1]:
            # APPENDS THE NUMBER GUESSED
            # losses.append(order[output.index(max(output))])
            # APPENDS THE EXPECTED NUMBER
            losses.append(img[1])
    # PRINT HOW MUCH EVERY NUMBER "FAILED"
    # nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # for num in nums:
    #     print("FAIL[" + str(num) + "]: ", losses.count(num))
    return round(100 - len(losses) * 100 / len(testing_data))

# EVALUATE MODEL ACCURACY
# weights, biases = load_weights_and_biases("weights_biases.json")
# print(eval_accuracy(weights, biases))
