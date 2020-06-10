import numpy as np
import itertools
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def sigmoid(x):
    return 1 /(1 + np.exp(-x))


def derivative_sigmoid(x):
    return sigmoid(x) / (1 - sigmoid(x))


def get_cost_value(Y_hat, Y):
        # number of examples
        m = Y_hat.shape[1]
        # calculation of the cost according to the formula
        cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
        return np.squeeze(cost)


class NeuralNetwork:
    def __init__(self, layers):
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layers[:-1], layers[1:])] 
    
    def feedforward(self, X):
        
        history = {}

        for idx, w in enumerate(self.weights):
            if idx == 0:
                a = np.dot(X, self.weights[idx].T)
                z = sigmoid(a)
                history[idx] = {'a': a, 'z': z}
            else:
                a = np.dot(z, self.weights[idx].T)
                z = sigmoid(a)
                history[idx] = {'a': a, 'z': z}

        return history


    def train(self, epochs=1, plot_data=True):
       
        # Cereate dataset
        X, y = make_circles()
        X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.33, random_state=42)
        
        
        # Plot what we are going to model
        if plot_data:
            plt.scatter(x=X[:,0], y=X[:,1], label=y)
            plt.show()


        # Feedforward
        history = self.feedforward(X_train)

        # Get cost value
        Y = y_train.reshape(y_train.shape[0], 1)
        cost_val  = get_cost_value(history[2]['z'], Y) 
        import ipdb; ipdb.set_trace() # BREAKPOINT
         
        
        # Calculate the Gradient


        
        pass
    


if __name__ == "__main__":
    NeuralNetwork([2, 16, 16, 1]).train(plot_data=False)
    print(test(2, 5))
