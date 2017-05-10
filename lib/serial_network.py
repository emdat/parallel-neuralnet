"""
serial_network.py
~~~~~~~~~~
A module to implement serial minibatch gradient descent learning
algorithm for a feedforward neural network.  

Serial version adapted using:
1) Denny Britz's tutorial on implementing a neural network
Link to tutorial: http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
Link to file in Github: https://github.com/dennybritz/nn-from-scratch/blob/master/ann_classification.py 
2) mnielsen's neural-networks-and-deep-learning reposity on GitHub, 
file src/network.py. 
Link to file: https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_loader.py
"""

import random
import numpy as np
import time 

def shuffle(X, y):
    p = np.random.permutation(len(X))
    return X[p], y[p]

class serial_network(object):

    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.weights = []#np.empty(self.num_layers, dtype=np.ndarray)
        self.biases = []#np.empty(self.num_layers, dtype=np.ndarray) 
        for i in range(1, self.num_layers):
            self.weights.append(np.random.randn(self.layer_dims[i-1], self.layer_dims[i]) / np.sqrt(self.layer_dims[i-1]))
            self.biases.append(np.zeros((self.layer_dims[i])))
    
    def train(self, train_data, num_epochs, mini_batch_sz=200, learning_rate=0.01, test_data=None):
        X = train_data[0]
        y = train_data[1]
        num_examples = len(X)
        self.sgd(X, y, num_examples, num_epochs, test_data, mini_batch_sz, learning_rate)
    
    def sgd(self, X, y, num_examples, num_epochs, test_data, mini_batch_sz, learning_rate, reg_lambda=0.01):
        if test_data: 
            n_test = len(test_data[0])
       
        tot_eval_time = 0.0
        start_sgd_time = time.clock()
        for epoch in xrange(num_epochs):
            X, y = shuffle(X, y)
            mini_batches_x = [X[k:k+mini_batch_sz] for k in xrange(0, num_examples, mini_batch_sz)]
            mini_batches_y = [y[k:k+mini_batch_sz] for k in xrange(0, num_examples, mini_batch_sz)]
            for mb_x, mb_y in zip(mini_batches_x, mini_batches_y):
                self.update_mini_batch(mb_x, mb_y, mini_batch_sz, learning_rate, reg_lambda)
        
            start_eval_time = time.clock()
            if test_data: 
                print "Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data[0], test_data[1]), n_test)
            end_eval_time = time.clock()
            tot_eval_time += end_eval_time - start_eval_time

        end_sgd_time = time.clock()

        # Print timing data
        print "Serial | {0} seconds for {1} epochs, {2} examples, {3} batch_sz".format(
                                                        end_sgd_time - start_sgd_time - tot_eval_time, 
                                                        num_epochs,
                                                        num_examples, mini_batch_sz) 
        if test_data:
            print "{0} seconds for evaluation.".format(tot_eval_time) 

    def update_mini_batch(self, x, y, mini_batch_sz, learning_rate, reg_lambda):
        self.fwd_prop(x)
        # Back propagation
        delta_weights = [None] * (self.num_layers-1)
        delta_biases = [None] * (self.num_layers-1)
        for l in range(1, self.num_layers):
            if l == 1:
                err = self.activations[-1]
                err[range(mini_batch_sz), y] -= 1
            else: 
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
        
        for (b, w) in zip(self.biases, self.weights):
            self.zs.append(self.activations[-1].dot(w) + b)
            self.activations.append(np.tanh(self.zs[-1]))
        
        self.activations[-1] = self.softmax(self.zs[-1])
    
    def softmax(self, x):
        # Generate probabilties
        e_x = np.exp(x - np.max(x)) # Normalize to prevent overflow
        return e_x / np.sum(e_x, axis=1, keepdims=True)
        
    def evaluate(self, X, y):
        """Return the number of test inputs for which the neural
        network outputs the correct classification."""
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
    
