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
        z = x
        for idx, l in enumerate(self.weights):

            # First linear transformation and than apply activation function
            z = (sigmoid(z.dot(self.weights[idx])))
            layers.append(np.expand_dims(z, axis=0))

        return layers

    def back_propagation(self, layers: list, x, y):

        # input layer + hidden layer(s) + output layer
        layers = [np.expand_dims(x, axis=0)] + layers
        for idx, w in enumerate(reversed(self.weights)):

            if idx == 0:
                output_error = y - layers[-1]
                delta = output_error * derivative_sigmoid(layers[-1])
            else:
                delta = delta.dot(self.weights[-(idx)].T) * derivative_sigmoid(layers[-(idx + 1)])

            # print(delta.shape)
            adjustment = layers[-(idx + 2)].T.dot(delta)
            self.weights[-(idx + 1)] += adjustment

        return sum(output_error**2).sum()

    def back_propagate(self, x, y):

        layers = self.forward(x)
        return self.back_propagation(layers=layers, x=x, y=y)

        # layer1 = np.expand_dims(sigmoid(x.dot(self.weights[0])), axis=0)
        # layer2 = sigmoid(layer1.dot(self.weights[1]))
        #
        # outputError = y - layer2
        # delta = outputError * derivative_sigmoid(layer2)
        # self.weights[1] += layer1.T.dot(delta)
        #
        # delta = delta.dot(self.weights[1].T) * derivative_sigmoid(layer1)
        # w1_adj = np.expand_dims(x, axis=0).T.dot(delta)
        # self.weights[0] += w1_adj
        #
        # return sum(outputError**2).sum()

    def train(self, X_train, y_train, epochs=1000, plot_data=True):

        for epoch in range(epochs):
            loss = []
            for idx in range(len(X_train)):
                l = self.back_propagate(x=X_train[idx], y=y_train[:, idx])
                loss.append(l)
            print(sum(loss))


if __name__ == "__main__":

    X, y = make_classification(
        n_samples=50, n_features=10, n_classes=2, n_informative=2
    )
    y = np.expand_dims(y, axis=0)
    nn = NeuralNetwork([10, 5, 8, 7, 1])
    nn.train(X_train=X, y_train=y)
