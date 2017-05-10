"""
parallel_network.py
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
import copy
import numpy as np
#from sklearn import datasets, linear_model
#from sklearn.utils import shuffle
from mpi4py import rc
rc.initialize = False
from mpi4py import MPI
from operator import add
nfetch = 5
npush = 5


def shuffle(X, y):
    p = np.random.permutation(len(X))
    return X[p], y[p]

def add_accrued(x, y): 
    return [tot + inc for tot, inc in zip(x, y)]

def add_grad(x, y):
    learning_rate = 3.0
    mini_batch_sz = 10
    # Gradient descent
    #i = 0
    #for w, dw in zip(x, y):
    #    x[i] = w-(learning_rate/mini_batch_sz)*dw
    #    i += 1
    x += -(learning_rate/mini_batch_sz)*y 
    #x = [w-(learning_rate/mini_batch_sz)*dw for w, dw in zip(x, y)]
    return x

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

class parallel_network(object):

    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.weights = []
        self.biases = [] 
        for i in range(1, self.num_layers):
            self.weights.append(np.random.randn(self.layer_dims[i-1], self.layer_dims[i]) / np.sqrt(self.layer_dims[i-1]))
            self.biases.append(np.zeros((self.layer_dims[i])))
        self.weights = np.asarray(self.weights)
        self.biases = np.asarray(self.biases)    

    def train(self, train_data, num_epochs, mini_batch_sz=200, learning_rate=0.01, test_data=None):
        X = train_data[0]
        y = train_data[1]
        num_examples = len(X)
        self.sgd(X, y, num_examples, num_epochs, test_data, mini_batch_sz, learning_rate)
    
    def sgd(self, X, y, num_examples, num_epochs, test_data, mini_batch_sz, learning_rate, reg_lambda=0.01, npf=1):
        if test_data: 
            n_test = len(test_data[0])
      
        MPI.Init()
        comm = MPI.COMM_WORLD
        nprocs = comm.Get_size()
        rank   = comm.Get_rank()
      
        num_workers = nprocs
        num_ex_per_worker = num_examples/num_workers
        batches_per_worker = num_ex_per_worker/mini_batch_sz
        leftover_ex = num_examples % num_workers 
        # Master sends subset of training data to workers
        X_per_worker = []
        y_per_worker = []
        if rank == 0:
            X_per_worker = [X[k:k+num_ex_per_worker] for k in xrange(0, num_examples-leftover_ex, num_ex_per_worker)]       
            y_per_worker = [y[k:k+num_ex_per_worker] for k in xrange(0, num_examples-leftover_ex, num_ex_per_worker)]       

            # Allocate any leftover examples to the last worker
            if leftover_ex > 0:
                X_per_worker[num_workers-1] = X[num_ex_per_worker*(num_workers - 1):num_examples] 
                y_per_worker[num_workers-1] = y[num_ex_per_worker*(num_workers - 1):num_examples] 
        
        # Receive this worker's X and y elements
        my_X = comm.scatter(X_per_worker, root=0)
        my_y = comm.scatter(y_per_worker, root=0)
        
        accrued_dw = [0] * (self.num_layers-1)
        accrued_db = [0] * (self.num_layers-1)
        
        # Do sgd for each epoch
        for epoch in xrange(num_epochs):
            print "hi"
            #if rank == 0:
                #if epoch > 0 and test_data:
                    #print "Epoch {0}: {1} / {2}".format(epoch-1, self.evaluate(test_data[0], test_data[1]), n_test)
                
            my_X, my_y = shuffle(my_X, my_y)
            my_leftover_ex = num_ex_per_worker % mini_batch_sz
            mini_batches_x = [my_X[k:k+mini_batch_sz] for k in xrange(0, num_ex_per_worker-my_leftover_ex, mini_batch_sz)]
            mini_batches_y = [my_y[k:k+mini_batch_sz] for k in xrange(0, num_ex_per_worker-my_leftover_ex, mini_batch_sz)] 
            
            step = 0
            old_weights = copy.deepcopy(self.weights) 
            old_biases = copy.deepcopy(self.biases) 
    
            for mb_x, mb_y in zip(mini_batches_x, mini_batches_y):
                delta_w, delta_b = self.update_mini_batch(mb_x, mb_y, mini_batch_sz, learning_rate, reg_lambda)

                # Gradient descent
                self.weights = [w-(learning_rate/mini_batch_sz)*dw for w, dw in zip(self.weights, delta_w)]
                self.biases = [b-(learning_rate/mini_batch_sz)*db for b, db in zip(self.biases, delta_b)]   
                
                accrued_dw = [acdw + dw for acdw, dw in zip(accrued_dw, delta_w)]
                accrued_db = [acdb + db for acdb, db in zip(accrued_db, delta_b)]
                if step % npf == 0:
                    accrued_dw = comm.allreduce(sendobj=accrued_dw, op=add_accrued)                     
                    accrued_db = comm.allreduce(sendobj=accrued_db, op=add_accrued)                     
                    # Apply updates
                    self.weights = [w-(learning_rate/mini_batch_sz)*dw for w, dw in zip(old_weights, accrued_dw)]
                    self.biases = [b-(learning_rate/mini_batch_sz)*db for b, db in zip(old_biases, accrued_db)]   
                    # Reset variables
                    accrued_dw = [0] * (self.num_layers-1)
                    accrued_db = [0] * (self.num_layers-1)
                    old_weights = self.weights#copy.deepcopy(self.weights) 
                    old_biases = self.biases#copy.deepcopy(self.biases) 

                step += 1
            #print "Epoch {0} complete".format(epoch)
    
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
        
        return delta_weights, delta_biases     
    
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
    

