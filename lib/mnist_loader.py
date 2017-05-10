"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.
From mnielsen's neural-networks-and-deep-learning reposity on GitHub, 
file src/mnist_loader.py. 
Link to file: https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_loader.py
"""

#### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper(num_train=None, num_test=None, num_val=None):
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    Each dataset is a list/array for X (the signal) and y (the 
    classification) separately."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, 784) for x in tr_d[0]]
    training_data = np.asarray(training_inputs), np.asarray(tr_d[1])
    validation_inputs = [np.reshape(x, 784) for x in va_d[0]]
    validation_data = np.asarray(validation_inputs), np.asarray(va_d[1])
    test_inputs = [np.reshape(x, 784) for x in te_d[0]]
    test_data = np.asarray(test_inputs), np.asarray(te_d[1])
    if num_train:
        training_data = training_data[:num_train]
    if num_test:
        test_data = test_data[:num_test]
    if num_val:
        validation_data = validation_data[:num_val]
    return (training_data, validation_data, test_data)
