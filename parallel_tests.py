#from sklearn import datasets, linear_model
#from sklearn.utils import shuffle
#import matplotlib.pyplot as plt
import numpy as np 
from lib import mnist_loader, parallel_network
from mpi4py import rc
rc.initialize = False
from mpi4py import MPI

def mnist():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    #print training_data[1]
    nnet = parallel_network.parallel_network([784, 100, 100, 100,  10])
    #def train(self, train_data, num_epochs, mini_batch_sz=200, learning_rate=0.01, test_data=None):
    nnet.train(training_data, 30, 10, 0.01, test_data)

mnist()
#moons()

