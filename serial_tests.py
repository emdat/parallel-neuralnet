import numpy as np 
from lib import mnist_loader, serial_network

def mnist():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    #print training_data[1]
    nnet = serial_network.neural_network([784, 30, 70, 20,  10])
    #def train(self, train_data, num_epochs, mini_batch_sz=200, learning_rate=0.01, test_data=None):
    nnet.train(training_data, 30, 10, 0.01, test_data)

mnist()
#moons()

