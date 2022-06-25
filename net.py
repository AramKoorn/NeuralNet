import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Nice explanation of the analytical formula's: https://datascience.stackexchange.com/questions/30676/role-derivative-of-sigmoid-function-in-neural-networks


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, layers):
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(layers[:-1], layers[1:])]

    def forward(self, x):

        layers = []
        for idx, l in enumerate(self.weights):

            # First linear transformation and than apply activation function
            x = sigmoid(x.dot(self.weights[idx]))
            layers.append(np.expand_dims(x, axis=0))

        return layers

    def back_propagation(self, layers: list, x, y):

        # input layer + hidden layer(s) + output layer
        layers = [np.expand_dims(x, axis=0)] + layers
        for idx, w in enumerate(reversed(self.weights)):

            if idx == 0:
                output_error = y - layers[-1]
                delta = output_error * derivative_sigmoid(layers[-1])
            else:
                delta = delta.dot(self.weights[-(idx)].T) * derivative_sigmoid(
                    layers[-(idx + 1)]
                )

            # print(delta.shape)
            adjustment = layers[-(idx + 2)].T.dot(delta)
            self.weights[-(idx + 1)] += adjustment

        return sum(output_error**2).sum()

    def train(self, X_train, y_train, epochs=1000, plot_data=True):

        for epoch in range(epochs):
            loss = []
            for idx in range(len(X_train)):
                layers = self.forward(x=X_train[idx])
                l = self.back_propagation(
                    layers=layers, x=X_train[idx], y=y_train[:, idx]
                )
                loss.append(l)
            print(sum(loss))


if __name__ == "__main__":

    X, y = make_classification(
        n_samples=50, n_features=10, n_classes=2, n_informative=2
    )
    y = np.expand_dims(y, axis=0)
    nn = NeuralNetwork([10, 5, 8, 7, 1])
    nn.train(X_train=X, y_train=y)
