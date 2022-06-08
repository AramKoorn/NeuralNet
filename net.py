import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Nice explanation of the analytical formula's: https://datascience.stackexchange.com/questions/30676/role-derivative-of-sigmoid-function-in-neural-networks


def create_classifier():
    # Cereate dataset
    X, y = make_circles()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    return X_train, X_test, y_train, y_test


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, layers):
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(layers[:-1], layers[1:])]

    def back_propagate(self, x, y):

        layer1 = np.expand_dims(sigmoid(x.dot(self.weights[0])), axis=0)
        layer2 = sigmoid(layer1.dot(self.weights[1]))

        outputError = y - layer2
        delta = outputError * derivative_sigmoid(layer2)
        self.weights[1] += layer1.T.dot(delta)

        delta = delta.dot(self.weights[1].T) * derivative_sigmoid(layer1)
        w1_adj = np.expand_dims(x, axis=0).T.dot(delta)
        self.weights[0] += w1_adj

        return sum(outputError**2).sum()

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
    nn = NeuralNetwork([10, 5, 1])
    nn.train(X_train=X, y_train=y)
    nn.weights[1].shape
