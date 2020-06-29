import numpy as np
import itertools
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def relu(x):
    return np.max(0, x)


def derivative_relu(x):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;


def sigmoid(x):
    return 1 /(1 + np.exp(-x))


def derivative_sigmoid(dA, Z):
        
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)


# Functioni crossentropy
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
        self.input_layer = None
    
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

    def back_propagate(self, history, Y, Y_hat):
        
        # Gradients we need to calculate 
        # dW 
        # db
        # dA
        # dZ

        m = Y.shape[0] 
        dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
        for idx, layer in enumerate(reversed(history.keys())):

            # print(f'Layer: {layer}')
            dA_curr = dA_prev
            da_act_func = derivative_sigmoid(dA_curr, history[layer]['z'].T)  # dZ
            if layer != 0:
                da_w = np.dot(history[layer - 1]['a'].T, da_act_func.T).T
            else:
                da_w = np.dot(self.input_layer.T, da_act_func.T).T 
            d_b = 1 / m * np.sum(da_act_func, keepdims=True)  # db
            dA_prev = np.dot(self.weights[layer].T, da_act_func) # dA_prev 

            # Update weights and biases  
            learning_rate = 0.01
            
            self.weights[layer] = self.weights[layer] - learning_rate * da_w
            self.biases[layer] = self.biases[layer] - learning_rate * d_b 
            
            # import ipdb; ipdb.set_trace() # BREAKPOINT
            
            # print(f'Weights are {self.weights}') 
            # print(f'biases are {self.biases}') 
        # print('succesfully iterated')


        pass        
        

    def train(self, epochs=1000, plot_data=True):
       
        # Cereate dataset
        X, y = make_circles()
        X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.33, random_state=42)
        self.input_layer = X_train
       
        print(f'Shape of X train: {X_train.shape}')
        
        # Plot what we are going to model
        if plot_data:
            plt.scatter(x=X[:,0], y=X[:,1], label=y)
            plt.show()

        for i in range(epochs):


            # Feedforward
            history = self.feedforward(X_train)
            if (history[2]['z'] == np.nan).sum() > 0:
                # print(i)
                break

            # if epochs % 100 == 0:
                # print(history[2]['z'])

            # Get cost value
            Y_hat = history[2]['z'].T
            predict = np.where(Y_hat > 0.5, 1, 0)
            precision = (predict == y_train).sum() / y_train.shape[0]
            print(f"Precision : {precision}")
            import ipdb; ipdb.set_trace() # BREAKPOINT

            cost_val  = get_cost_value(Y_hat, y_train) 
            # import ipdb; ipdb.set_trace() # BREAKPOINT
             
            
            # backward propagation
            gradients = self.back_propagate(history, y_train, Y_hat)
        
        
        pass
    


if __name__ == "__main__":
    NeuralNetwork([2, 16, 16, 1]).train(plot_data=False)
