# DigitRecon
A custom neural network written from scratch in Python with no external libraries (only exception being pillow, used for converting images to pixel arrays).

The model takes in a 10x10 image representing any digit as an input, and outputs the most likely digit to be represented in the image. The whole architecture was written in less than a day in Python, but is not meant to be optimized at all. The goal of this project was to challenge myself and see how fast I could write a custom neural network from scratch, with no prior experience in machine learning.

### Architecture
The model has 4 layers:
- An input layer consisting of 100 neurons, representing a 10x10 image (thus a total of 100 pixels, one neuron per pixel). The activation of a neuron in the input layer will be 1 if the corresponding pixel is part of the digit's shape to indentify, else it will be 0
- Two hidden layers each consisting of 81 neurons
- An output layer, where the most active neuron will be the recognized digit

### Dataset
The model utilizes a custom dataset I have made myself using a custom pixel art drawer created with Pygame. The images in the dataset are black and white and pixel-perfect, thus making the model bad at recognizing very blurry / colorful images.
#### Using your own dataset
You can use your own dataset (or add your own images to the existing one) as long as you modify the img_to_pixels function to convert your images to the format required that is specified in the input layer's architecture.

### Training
The model is currently trained with a learning rate of 0.01 across 1000 epochs. I heavily recommend the usage of PyPy when running train.py to substantially speed up this process.

### Accuracy
The model is not very accurate as of now (~78%) due to a lack of data, even though the dataset is constantly growing. 

### Performance
The model is quite fast, taking less than 40 milliseconds in total to read the weights, run the model, and print the result.