import numpy as np
import itertools
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split


def sigmoid(x):
    return 1 /(1 + np.exp(-x))


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


    def train(self, epochs=1):
       
        # Cereate dataset
        X, y = make_circles()
        X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.33, random_state=42)

        # Feedforward
        history = self.feedforward(X)



        import ipdb; ipdb.set_trace() # BREAKPOINT
        
        pass
    


if __name__ == "__main__":
    NeuralNetwork([2, 16, 16, 1]).train()
    print(test(2, 5))
