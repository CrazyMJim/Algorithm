from numpy import exp
from numpy import log
from numpy import ndarray
from numpy import sum
import scipy.optimize as opt
import numpy as np


def sigmoid(z):
    return 1 / (1 + exp(-z))


def cost(theta, x, y):
    z = theta.transpose().dot(x.transpose())
    h = sigmoid(z)
    j = sum(-y.transpose() * log(h) - (1 - y).transpose() * log(1 - h)) / len(h)
    return j


def gradient(theta, x, y):
    z = theta.transpose().dot(x.transpose())
    h = sigmoid(z)
    error = h - y.transpose()
    grad = (error.dot(x) / len(h)).transpose()
    return grad


def cost_reg(theta, x, y, rate):
    z = theta.transpose().dot(x.transpose())
    h = sigmoid(z)
    tmp_theta = theta[1:3]
    tmp_theta.shape = (2, 1)
    j = sum(-y.transpose() * log(h) - (1 - y).transpose() * log(1 - h)) / len(h)
    j += sum(pow(tmp_theta, 2)) * rate / (2 * len(h))
    return j


def gradient_reg(theta, x, y, rate):
    z = theta.transpose().dot(x.transpose())
    h = sigmoid(z)
    error = h - y.transpose()
    tmp_theta = theta[1:3]
    tmp_theta.shape = (2, 1)
    tmp_theta = np.row_stack((np.zeros((1, 1)), tmp_theta)) * rate / len(h)
    grad = (error.dot(x) / len(h)).transpose()
    grad.shape = (3, 1)
    grad += tmp_theta
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
        j = cost(result[0], x, y)
        print("Cost:", end="")
        print(j)
        return result[0]


def train_reg(data_set):
    if isinstance(data_set, ndarray):
        m, n = data_set.shape
        data_set = np.column_stack((np.ones((m, 1)), data_set))
        m, n = data_set.shape
        theta = np.zeros((n-1, 1))
        x = data_set[:, 0:n-1]
        y = data_set[:, n-1]
        rate = 0.01
        result = opt.fmin_tnc(func=cost_reg, x0=theta, fprime=gradient_reg, args=(x, y, rate))
        print(result)

