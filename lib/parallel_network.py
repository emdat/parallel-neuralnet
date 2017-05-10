"""
parallel_network.py
~~~~~~~~~~
A module to implement parallel minibatch gradient descent learning
algorithm for a feedforward neural network. 

The serial version of this code was somewhat adapted from:
1) Denny Britz's tutorial on implementing a neural network
Link to tutorial: http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
Link to file in Github: https://github.com/dennybritz/nn-from-scratch/blob/master/ann_classification.py 
2) mnielsen's neural-networks-and-deep-learning reposity on GitHub, 
file src/network.py. 
Link to file: https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_loader.py
"""

import time
import random
import copy
import numpy as np
from mpi4py import rc
rc.initialize = False
from mpi4py import MPI
from operator import add

# Shuffle lists X and y in unison
def shuffle(X, y):
    p = np.random.permutation(len(X))
    return X[p], y[p]

# Add the two lists element-wise
def add_accrued(x, y): 
    return [tot + inc for tot, inc in zip(x, y)]

class parallel_network(object):

    # Create an instance of a parallel neural network
    def __init__(self, layer_dims, reg_lambda=0.01, npf=1):
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.reg_lambda = reg_lambda
        self.npf = npf
        self.weights = []
        self.biases = [] 
        for i in range(1, self.num_layers):
            self.weights.append(np.random.randn(self.layer_dims[i-1], self.layer_dims[i]) / np.sqrt(self.layer_dims[i-1]))
            self.biases.append(np.zeros((self.layer_dims[i])))
        self.weights = np.asarray(self.weights)
        self.biases = np.asarray(self.biases)    

    # Train the network on the training data given, which has the input data as a its first attribute
    # and the inputs' correct classifications as its second attribute. 
    def train(self, train_data, num_epochs, mini_batch_sz, learning_rate=0.01, test_data=None):
        X = train_data[0]
        y = train_data[1]
        num_examples = len(X)
        MPI.Init()
        self.sgd(X, y, num_examples, num_epochs, test_data, mini_batch_sz, learning_rate)
        MPI.Finalize()
   
    # Conduct minibatch (stochastic) gradient descent on the training data given as X, y
    def sgd(self, X, y, num_examples, num_epochs, test_data, mini_batch_sz, learning_rate):
        if test_data: 
            n_test = len(test_data[0])
      
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
        
        comm.Barrier()
        tot_communic_time = 0.0
        tot_eval_time = 0.0
        if rank == 0: # Start timer for entirety of sgd 
            start_sgd_time = time.clock()
        
        # Do sgd for each epoch
        for epoch in xrange(num_epochs):
            
            # Randomize the examples. Ensure all processes have the same number of minibatches 
            # (necessary for correct coordination). 
            my_X, my_y = shuffle(my_X, my_y)
            my_leftover_ex = num_ex_per_worker % mini_batch_sz
            mini_batches_x = [my_X[k:k+mini_batch_sz] for k in xrange(0, num_ex_per_worker-my_leftover_ex, mini_batch_sz)]
            mini_batches_y = [my_y[k:k+mini_batch_sz] for k in xrange(0, num_ex_per_worker-my_leftover_ex, mini_batch_sz)] 
            
            step = 0
            old_weights = copy.deepcopy(self.weights) 
            old_biases = copy.deepcopy(self.biases) 
    
            # Iterate through the minibatches
            for mb_x, mb_y in zip(mini_batches_x, mini_batches_y):
                delta_w, delta_b = self.update_mini_batch(mb_x, mb_y, mini_batch_sz, learning_rate)

                # Gradient descent
                self.weights = [w-(learning_rate/mini_batch_sz)*dw for w, dw in zip(self.weights, delta_w)]
                self.biases = [b-(learning_rate/mini_batch_sz)*db for b, db in zip(self.biases, delta_b)]   
                
                accrued_dw = [acdw + dw for acdw, dw in zip(accrued_dw, delta_w)]
                accrued_db = [acdb + db for acdb, db in zip(accrued_db, delta_b)]
                
                if step % self.npf == 0:
                    start_communic_time = time.clock()
                    accrued_dw = comm.allreduce(sendobj=accrued_dw, op=add_accrued)                     
                    accrued_db = comm.allreduce(sendobj=accrued_db, op=add_accrued)                     
                    end_communic_time = time.clock()
                    tot_communic_time += end_communic_time - start_communic_time
                    # Apply updates
                    self.weights = [w-(learning_rate/mini_batch_sz)*dw for w, dw in zip(old_weights, accrued_dw)]
                    self.biases = [b-(learning_rate/mini_batch_sz)*db for b, db in zip(old_biases, accrued_db)]   
                    # Reset variables
                    accrued_dw = [0] * (self.num_layers-1)
                    accrued_db = [0] * (self.num_layers-1)
                    old_weights = copy.deepcopy(self.weights) 
                    old_biases = copy.deepcopy(self.biases) 
            
            # Evaluate the parameters on the test data. We will exclude this from our overall timings. 
            if test_data: 
                comm.Barrier()
                start_eval_time = time.clock()
                if rank == 0:
                    print "Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data[0], test_data[1]), n_test)
                comm.Barrier()
                stop_eval_time = time.clock()
                if rank==0:
                    tot_eval_time += stop_eval_time - start_eval_time

                step += 1
        
        comm.Barrier()
        # Print timings
        if rank == 0:
            end_sgd_time = time.clock()
            print "{0} proccesses | {1} seconds for {2} epochs, {3} examples, {4} batch_sz".format(
                                                                        nprocs, 
                                                                        end_sgd_time - start_sgd_time - tot_eval_time, 
                                                                        num_epochs,
                                                                        num_examples,
                                                                        mini_batch_sz)
            if test_data:
                print "{0} seconds for evaluation.".format(tot_eval_time) 
        print "\tRank {0}: {1} seconds on communications".format(rank, tot_communic_time)

    def update_mini_batch(self, x, y, mini_batch_sz, learning_rate):
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
            delta_weights[-l] += self.reg_lambda * self.weights[-l]    
        
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
        return e_x / np.sum(e_x, axis=1, keepdims=True)
        
    def evaluate(self, X, y):
        """Return the total number of test inputs for which the neural
        network outputs the correct classification result.""" 
        test_results = []
        for x, y in zip(X, y):
            self.fwd_prop(x)
            probs = self.activations[-1]
            test_results.append((np.argmax(probs, axis=1), y))
        return sum(int(xx == yy) for (xx, yy) in test_results)

    # Predict the classification for input x. 
    def predict(self, x):
        self.fwd_prop(x)
        return np.argmax(self.activations[-1], axis=1)
