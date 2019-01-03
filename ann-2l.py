import numpy as np

data = np.array([[1, 0, 1],
                 [0, 1, 0],
                 [1, 0, 0],
                 [0, 1, 1]])
target = np.array([[1, 0, 1, 0]]).T


def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return (1 / (1 + np.exp(-x)))


# 2 layers,
# first layer is input with 3 nodes/features,
# second layer is output with 1 node
def train(n, learning_rate):
    # Initialize random seed and weights
    np.random.seed(1)
    # Initialized weights to be random between [-1, 1)
    w0 = 2 * np.random.random((3, 1)) - 1
    for i in range(n):
        layer0 = data
        layer1 = sigmoid(layer0.dot(w0))
        error = layer1 - target
        delta = error * sigmoid(layer1, deriv=True)
        w0 -= learning_rate * layer0.T.dot(delta)
        if i % 1000 == 0:
            print('Weights: ', w0)
            print('Error: ', error)
            print('Results: ', layer1)


if __name__ == '__main__':
    train(20000, 10)
