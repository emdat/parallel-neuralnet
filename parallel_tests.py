import numpy as np 
from lib import mnist_loader, parallel_network
from mpi4py import rc
rc.initialize = False
from mpi4py import MPI
import sys

def mnist(num_train=None, mini_batch_size=10, num_epochs=30):
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper(num_train=num_train)
    nnet = parallel_network.parallel_network([784, 30,  10])
    #def train(self, train_data, num_epochs, mini_batch_sz=200, learning_rate=0.01, test_data=None):
    nnet.train(training_data, num_epochs, mini_batch_sz, 0.01, test_data)

np.random.seed(0)
num_train = None
mini_batch_sz = 10
num_epochs=30
if len(sys.argv) > 1:
    num_train = int(sys.argv[1])
if len(sys.argv) > 2:
    mini_batch_sz = int(sys.argv[2])
if len(sys.argv) > 3:
    num_epochs = int(sys.argv[3])
mnist(num_train, mini_batch_sz, num_epochs)

