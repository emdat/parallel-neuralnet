"""
neural_network.py
~~~~~~~~~~
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  

Largely adapted from:
1) Denny Britz's tutorial on implementing a neural network
Link to tutorial: http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
Link to file in Github: https://github.com/dennybritz/nn-from-scratch/blob/master/ann_classification.py 
2) mnielsen's neural-networks-and-deep-learning reposity on GitHub, 
file src/network.py. 
Link to file: https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_loader.py
"""

import random
import numpy as np
from sklearn import datasets, linear_model
from sklearn.utils import shuffle

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

class neural_network(object):

    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.weights = []
        self.biases = [] 
        for i in range(1, self.num_layers):
            self.weights.append(np.random.randn(self.layer_dims[i-1], self.layer_dims[i]) / np.sqrt(self.layer_dims[i-1]))
            self.biases.append(np.zeros((1, self.layer_dims[i])))
    
    def train(self, train_data, num_epochs, mini_batch_sz=200, learning_rate=0.01, test_data=None):
        X = train_data[0]
        y = train_data[1]
        num_examples = len(X)
        self.sgd(X, y, num_examples, num_epochs, test_data, mini_batch_sz, learning_rate)
    
    def sgd(self, X, y, num_examples, num_epochs, test_data, mini_batch_sz, learning_rate, reg_lambda=0.01):
        if test_data: 
            n_test = len(test_data[0])
        
        for epoch in xrange(num_epochs):
            X, y = shuffle(X, y, random_state=0)
            mini_batches_x = [X[k:k+mini_batch_sz] for k in xrange(0, num_examples, mini_batch_sz)]
            mini_batches_y = [y[k:k+mini_batch_sz] for k in xrange(0, num_examples, mini_batch_sz)]
            for mb_x, mb_y in zip(mini_batches_x, mini_batches_y):
                self.update_mini_batch(mb_x, mb_y, mini_batch_sz, learning_rate, reg_lambda)
            if test_data: 
                print "Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data[0], test_data[1]), n_test)
            #print "Epoch {0} complete".format(epoch)

    def update_mini_batch(self, x, y, mini_batch_sz, learning_rate, reg_lambda):
        self.fwd_prop(x)
        # Back propagation
        delta_weights = [None] * (self.num_layers-1)
        delta_biases = [None] * (self.num_layers-1)
        for l in range(1, self.num_layers):
            if l == 1:
                #err = self.activations[-1]-y
                #err = err * sigmoid_prime(self.zs[-1])
                err = self.activations[-1]
                err[range(mini_batch_sz), y] -= 1
            else: 
                #err = err.dot(self.weights[-l+1].T) * sigmoid_prime(self.zs[-1]) 
                err = err.dot(self.weights[-l+1].T) * (1 - np.square(self.activations[-l]))
            delta_weights[-l] = self.activations[-l-1].T.dot(err)
            delta_biases[-l] = np.sum(err, axis=0, keepdims=True)
            # Regularization
            delta_weights[-l] += reg_lambda * self.weights[-l]    
        
        # Gradient descent
        self.weights = [w-(learning_rate/mini_batch_sz)*dw for w, dw in zip(self.weights, delta_weights)]
        self.biases = [b-(learning_rate/mini_batch_sz)*db for b, db in zip(self.biases, delta_biases)]   
    
    # Forward propagation
    def fwd_prop(self, x):
        self.zs = []
        self.activations = []
        self.activations.append(x)
        #print type(self.activations[-1])
        #print self.activations[-1]
        #print self.weights[0] 
        for (b, w) in zip(self.biases, self.weights):
            self.zs.append(self.activations[-1].dot(w) + b)
            self.activations.append(np.tanh(self.zs[-1]))
            #self.activations.append(sigmoid(self.zs[-1]))
        
        self.activations[-1] = self.softmax(self.zs[-1])
    
    def softmax(self, x):
        # Generate probabilties
        e_x = np.exp(x - np.max(x)) # Normalize to prevent overflow
        #return np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
        return e_x / np.sum(e_x, axis=1, keepdims=True)
        
    def evaluate(self, X, y):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = []
        for x, y in zip(X, y):
            self.fwd_prop(x)
            probs = self.activations[-1]
            #print probs
            test_results.append((np.argmax(probs, axis=1), y))
            #print test_results[-1]
        return sum(int(xx == yy) for (xx, yy) in test_results)

    def predict(self, x):
        self.fwd_prop(x)
        return np.argmax(self.activations[-1], axis=1)
    
    # Helper function to evaluate the total loss on the datase
    def calculate_loss(model, X, y):
        num_examples = len(X)  # training set size
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        # Forward propagation to calculate our predictions
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Calculating the loss
        corect_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(corect_logprobs)
        # Add regulatization term to loss (optional)
        data_loss += Config.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1. / num_examples * data_loss
    

