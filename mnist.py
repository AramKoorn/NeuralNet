from net import NeuralNetwork
import numpy as np
import gzip
import matplotlib.pyplot as plt


def to_categorical(x):
    return np.array([1 if i == x else 0 for i in range(10)])


def fetch_mnist():
    parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
    X_train = parse("data/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).astype(np.float32)
    Y_train = parse("data/train-labels-idx1-ubyte.gz")[8:]
    X_test = parse("data/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).astype(np.float32)
    Y_test = parse("data/t10k-labels-idx1-ubyte.gz")[8:]
    return X_train, Y_train, X_test, Y_test


def main():

    X_train, y_train, X_test, y_test = fetch_mnist()

    model = NeuralNetwork([28 * 28, 10, 10], activation_output="softmax")
    y_train = np.array([to_categorical(x) for x in y_train])
    # Because we only use sigmoids it doesn't learn (quickly)
    model.train(X_train=X_train[:1000], y_train=y_train[:1000], epochs=100)

    pass


if __name__ == "__main__":
    main()