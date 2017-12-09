from numpy import exp
from numpy import log
from numpy import ndarray
import scipy.optimize as opt
import numpy as np


def sigmoid(z):
    return 1 / (1 + exp(-z))


def cost(theta, x, y):
    z = theta.transpose().dot(x.transpose())
    h = sigmoid(z)
    J = sum(-y.transpose() * log(h) - (1 - y).transpose() * log(1 - h)) / len(h)
    return J


def gradient(theta, x, y):
    z = theta.transpose().dot(x.transpose())
    h = sigmoid(z)
    error = h - y.transpose()
    grad = (error.dot(x) / len(h)).transpose()
    return grad


def train(data_set):
    if isinstance(data_set, ndarray):
        m, n = data_set.shape
        data_set = np.column_stack((np.ones((m, 1)), data_set))
        m, n = data_set.shape
        theta = np.zeros((n-1, 1))
        x = data_set[:, 0:n-1]
        y = data_set[:, n-1]
        result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(x, y))
        J = cost(result[0], x, y)
        print("Cost:", end="")
        print(J)
        return result[0]

