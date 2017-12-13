import numpy as np
from numpy import exp
from numpy import log
from numpy import ndarray
from numpy import sum


class LogisticRegression:
    def __init__(self):
        self.__n = 0
        self.__m = 0
        self.__x = None
        self.__y = None
        self.__lambdaP = 1
        self.__theta = None
        self.__theta = 0
        self.__cost_history = 0

    #训练数据前需要将数据无量纲化
    #本例使用吴恩达老师在cousera中使用的数据并进行标准化后得到损失值为0.25934
    def train(self, data_set):
        if isinstance(data_set, ndarray):
            m, n = data_set.shape
            data_set = np.column_stack((np.ones((m, 1)), data_set))
            m, n = data_set.shape
            self.__n = n - 1
            self.__m = m
            self.__theta = np.zeros((n-1, 1))
            self.__x = data_set[:, 0:n-1]
            self.__y = data_set[:, n-1]

            count = 0
            cost = -1
            while count < 1000 and self.__cost_history != cost:
                self.__cost_history = cost
                z = self.__theta.transpose().dot(self.__x.transpose())
                h = self.sigmoid(z)
                cost = self.cost_function(h)
                self.gradient_descent(h)
                count += 1
                print("Cost:", end="")
                print(cost)

            print(self.__theta)

    def sigmoid(self, z):
        return 1 / (1 + exp(-z))

    def gradient_descent(self, h):
        error = h - self.__y.transpose()
        self.__theta -= (error.dot(self.__x) / self.__m).transpose()

    def cost_function(self, h):
        cost = sum(-self.__y.transpose() * log(h) - (1 - self.__y).transpose() * log(1 - h)) / self.__m
        return cost

    def predict(self, x):
        probability = self.sigmoid(self.__theta.transpose() * x)
        if probability >= 0.5:
            return "1"
        else:
            return "0"
