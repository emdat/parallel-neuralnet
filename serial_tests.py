#from sklearn import datasets, linear_model
#from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np 
from lib_serial import mnist_loader
from lib_serial import neural_network as network

def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y


def visualize(X, y, model):
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    plot_decision_boundary(lambda x:model.predict(x), X, y)
    plt.title("Logistic Regression")


def plot_decision_boundary(pred_func, X, y):
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
    plt.show()

def moons():
    np.random.seed(0)
    X, y = generate_data()
    num_examples = len(X)
    nnet = network.neural_network([2, 3, 2])
    nnet.train((X, y), 20000, 10, 3.0, (X, y))
    visualize(X, y, nnet)

def mnist():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    #print training_data[1]
    nnet = network.neural_network([784, 30, 70, 20,  10])
    #def train(self, train_data, num_epochs, mini_batch_sz=200, learning_rate=0.01, test_data=None):
    nnet.train(training_data, 30, 10, 0.01, test_data)

mnist()
#moons()

