import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Nice explanation of the analytical formula's: https://datascience.stackexchange.com/questions/30676/role-derivative-of-sigmoid-function-in-neural-networks


def softmax(x):
    # Numerically stable with large exponentials
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, layers, activation_output="sigmoid"):
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(layers[:-1], layers[1:])]
        self.activation_output = activation_output

    def forward(self, x):
        '''
        :param x: input layer
        :return: list of tuples [(linear layer, activation layer)]
        '''

        layers = []
        for idx, l in enumerate(self.weights):

            # First linear transformation and than apply activation function
            z = x.dot(self.weights[idx])
            a = sigmoid(z)

            layers.append((np.expand_dims(z, axis=0), np.expand_dims(a, axis=0)))
            x = a

        return layers

    def back_propagation(self, layers: list, x, y):

        # input layer + hidden layer(s) + output layer
        layers = [(np.expand_dims(x, axis=0), np.expand_dims(x, axis=0))] + layers
        for idx, w in enumerate(reversed(self.weights)):

            if idx == 0:
                if self.activation_output == "sigmoid":
                    output_error = y - layers[-1][1]
                    delta = output_error * derivative_sigmoid(layers[-1][1])
                if self.activation_output == "softmax":
                    ...

            else:
                delta = delta.dot(self.weights[-(idx)].T) * derivative_sigmoid(
                    layers[-(idx + 1)][1]
                )

            # print(delta.shape)
            adjustment = layers[-(idx + 2)][1].T.dot(delta)
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
    nn.train(X_train=X, y_train=y, epochs=30)
