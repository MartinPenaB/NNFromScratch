import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')
m, n = data.shape
data = np.array(data).T

labels = data[0]
pixels = data[1:] / 255  # vertical vectors


def one_hot(labels):
    # Function to convert categorical labels into one-hot encoding

    one_hot = np.zeros((labels.size, 10))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot.T


def get_image(index):
    plt.imshow(np.reshape(pixels.T[index], (28, 28)))
    plt.show()


class Layer:
    # base class for all Layer in the network

    def forward(self, x):
        pass

    def backprop(self, g, learning_rate):
        pass


class Linear(Layer):

    # The Linear class represents learning_rate dense layer's linear transformation of the input

    def __init__(self, in_n, out_n):
        # in_n is the number of neurons in the input layer
        # out_n is the number of neurons in the output layer

        # Initialize weights and biases with random values between -0.5 and 0.5
        self.w = np.random.rand(out_n, in_n) - 0.5
        self.b = np.random.rand(out_n, 1) - 0.5

    def forward(self, x):
        # Store the input data as an attribute to be used later in backpropagation
        self.x = x

        # Calculate the dot product of the weights and input data, plus the bias
        return np.dot(self.w, x) + self.b

    def backprop(self, g, learning_rate):
        # g is the gradient from the next layer

        # Calculate the gradient for the previous layer
        gradient = np.dot(self.w.T, g)

        # Update the weights and biases using gradient descent
        self.w -= np.dot(g, self.x.T) * learning_rate / m
        self.b -= np.sum(g, axis=1, keepdims=True) * learning_rate / m

        # Return the gradient for the previous layer
        return gradient


class ReLU(Layer):

    # The ReLU class represents a Rectified Linear Unit activation function layer in a neural network

    def forward(self, x):
        # Store the input data as an attribute to be used later in backpropagation
        self.x = x

        # if input is less than 0 return 0 else input
        return np.maximum(x, 0)

    def backprop(self, g, learning_rate):
        # g is the gradient from the next layer
        # learning rate not used in this layer

        # Return the gradient for the previous layer
        return g * (self.x > 0)


class Softmax(Layer):

    # The Softmax class represents learning_rate Softmax activation function layer in learning_rate neural network

    def forward(self, x):
        # Normalize the input data to ensure stability of the exponentiation
        # n = (x - np.min(x)) / (np.max(x) - np.min(x))

        # Return the softmax of the input data
        # return np.exp(n) / np.sum(np.exp(n), axis=0)
        return np.exp(x) / sum(np.exp(x))

    def backprop(self, predictions, learning_rate):
        # predictions is the predicted output from forward propagation

        # Return the gradient for the previous layer
        # Already includes cross entropy loss and softmax
        return predictions - one_hot(labels)


network = [
    Linear(n - 1, n - 1),
    ReLU(),
    Linear(n - 1, 10),
    Softmax()
]


def get_accuracy(predictions, labels):
    return np.sum(predictions == labels) / labels.size


# train
print('training...')

for epoch in range(100):
    # Loop through 100 epochs

    x = pixels

    # Forward propagate the input data through the layers of the network
    for layer in network:
        x = layer.forward(x)

    predictions = np.argmax(x, axis=0)
    # Calculate loss using mean squared error
    # Cross entropy loss is used in the actual calculations
    loss = np.mean((labels - predictions) ** 2)

    accuracy = get_accuracy(predictions, labels)

    # plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epoch, accuracy, 'b-o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epoch, loss, 'b-o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()

    # Print the loss and accuracy along with the image and prediction/actual target information
    if epoch / 99 == 1:
        print('Predictions: ', predictions)
        print('Labels: ', labels)
        print('Accuracy:', accuracy)
        print('Loss: ', loss)
        plt.show()

        for i in range(epoch, epoch + 4):
            print('Prediction: ', predictions[i])
            print('Actual: ', labels[i])
            get_image(i)

    # Backpropagate the error through the layers of the network
    for layer in reversed(network):
        x = layer.backprop(x, 0.5)
