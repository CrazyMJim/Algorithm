import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from numpy import tanh

np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)
plt.scatter(X[:, 0], X[:, 1], s=40, c=y)

num_examples = len(X)
input_dim = 2
output_dim = 2
epsilon = 0.01
reg_lambda = 0.01

def nn_model(model, input_num):
    w1, w2, b1, b2 = model['w1'], model['w2'], model['b1'], model['b2']
    z1 = input_num.dot(w1) + b1
    a1 = tanh(z1)
    z2 = a1.dot(w2) + b2  # second layer neural network
    probs = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)  # softmax function
    return {'a1': a1, 'probs': probs}


def calculate_loss(model):
    result = nn_model(model, X)
    a2 = result['probs']
    # 计算损失
    # 因为两个神经元代表0跟1类别对应的概率，所以根据实际类别y，选出对应类别的神经元
    correct_logprobs = y * np.log(a2[range(0, num_examples), y])
    loss = -1 * np.sum(correct_logprobs) / num_examples
    return loss


def predict(model, x):
    result = nn_model(model, x)
    return np.argmax(result['probs'], axis=1)


def train(hdim, num_iters=2000, print_loss=False):
    np.random.seed(0)
    w1 = np.random.randn(input_dim, hdim)
    b1 = np.zeros((1, hdim))
    w2 = np.random.randn(hdim, output_dim)
    b2 = np.zeros((1, output_dim))

    model = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    learn_rate = 0.01

    for i in range(0, num_iters):
        result = nn_model(model, X)
        delta3 = result['probs']
        # 计算预测概率与1的误差
        delta3[range(num_examples), y] -= 1
        dw2 = result['a1'].T.dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = (1 - np.power(result['a1'], 2)) * delta3.dot(model['w2'].T)
        dw1 = X.T.dot(delta2)
        db1 = np.sum(delta2, axis=0)

        model['w1'] -= learn_rate * dw1
        model['w2'] -= learn_rate * dw2
        model['b1'] -= learn_rate * db1
        model['b2'] -= learn_rate * db2

        if print_loss and i % 1000:
            print("Loss after iteration %i: %f" %(i, calculate_loss(model)))

    return model


def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


model = train(3, 2001, print_loss=True)

plot_decision_boundary(lambda x: predict(model, x))
plt.title("Decision Boundary for hidden layer size 3")
plt.show()

