import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
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
    return sigmoid(x) / (1 - sigmoid(x))


class NeuralNetwork:
    def __init__(self, layers):
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

    def feedforward(self, X):

        a_list = []
        z_list = []

        a = X
        for idx, w in enumerate(self.weights):
            a = w.dot(a.T) + self.biases[idx].T
            z = sigmoid(a)

            a_list.append(a)
            a = z
            z_list.append(z)

        loss = z

        return a_list, z_list, loss

    def back_propagate(self, history, Y, Y_hat):

        # Gradients we need to calculate
        # dW
        # db
        # dA
        # dZ
        pass

    def train(self, X_train, y_train, epochs=1, plot_data=True,):

        for epoch in range(epochs):
            loss = []
            for idx in range(len(X_train)):
                a, z, output_layer = self.feedforward(X_train[idx])
                print(output_layer)

                # Calculate loss
                l = (output_layer - y_train[idx])**2
                loss.append(l)

                # Update weights by backpropagation
                self.back_propagate()


        # Feedforward
        history = self.feedforward(X_train)

        # Get cost value
        Y_hat = (history[2]["z"].T,)
        cost_val = get_cost_value(Y_hat, y_train)

        # backward propagation
        gradients = self.back_propagate(history, Y, Y_hat)

        pass


if __name__ == "__main__":

    # Create data
    X_train, X_test, y_train, y_test = create_classifier()
    plt.scatter(x=X_train[:, 0], y=X_train[:, 1], label=y_train)
    plt.show()

    nn = NeuralNetwork([2, 3, 1])
    nn.train(X_train=X_train, y_train=y_train)
    nn.weights[1].shape