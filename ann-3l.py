# -*- coding: utf-8 -*-
import numpy as np

# Signoid function with derivative


def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return (1 / (1 + np.exp(-x)))


def relu(x, deriv=False):
    if deriv:
        temp = x.copy()
        x[x <= 0] = 0
        return x
    return np.maximum(x, 0)


def train(N, features, hidden_layer, output_dim, epochs, l_rate, a_func, dropout_percent, dataset=1):

    # N is batch size; features is input dimension;
    # hidden_layer is hidden dimension; output_dim is output dimension.
    # Create random input and output data
    if dataset == 1:
        np.random.seed(1)
        data = 2 * np.random.rand(N, features) - 1  # random normalized input
        # random normalized output
        y_target = 2 * np.random.rand(N, output_dim) - 1
        # Randomly initialize weights
        w1 = np.random.randn(features, hidden_layer)
        w2 = np.random.randn(hidden_layer, output_dim)
    else:
        data = np.array([[0, 1, 0, 0] * (features // 4),
                         [1, 0, 1, 0] * (features // 4),
                         [1, 0, 0, 0] * (features // 4),
                         [0, 1, 1, 0] * (features // 4)])
        y_target = np.array([[0] * output_dim,
                             [1] * output_dim,
                             [1] * output_dim,
                             [0] * output_dim])
        w1 = 2 * np.random.rand(features, hidden_layer) - 1
        w2 = 2 * np.random.rand(hidden_layer, output_dim) - 1
    for t in range(epochs):
        # Forward pass: compute predicted y
        # Go to hidden layer
        l1 = data.dot(w1)
        # Relu as activation function
        l1_out = a_func(l1)
        # Apply dropout to turn off part of the hidden layer
        # and compensate the turned off by increase others
        l1_out *= np.random.binomial(
            [np.ones((len(data), hidden_layer))],
            1 - dropout_percent)[0] * (1.0 / (1 - dropout_percent)
                                       )
        l2 = l1_out.dot(w2)
        y_pred = a_func(l2)

        error = y_pred - y_target
        # Compute and print loss
        if t % 100 == 0:
            loss = np.square(error).sum()
            print(t, loss)
            # print(y_pred)

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * error * a_func(y_pred, deriv=True)
        grad_w2 = l1_out.T.dot(grad_y_pred)
        grad_h = grad_y_pred.dot(w2.T) * a_func(l1_out, deriv=True)
        grad_w1 = data.T.dot(grad_h)
        # Update weights
        w1 -= l_rate * grad_w1
        w2 -= l_rate * grad_w2


if __name__ == '__main__':
    train(64, 1000, 100, 10, 5000, 1, a_func=sigmoid, dropout_percent=0.5)
