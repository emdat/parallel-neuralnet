import numpy as np 
from lib import mnist_loader, serial_network
import sys 

def mnist(num_train=None, mini_batch_size=10, num_epochs=30):
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper(num_train=num_train)
    nnet = serial_network.serial_network([784, 30,  10])
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
