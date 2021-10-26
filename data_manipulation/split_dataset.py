from numpy.random import default_rng
import numpy as np
from data_manipulation.load_dataset import *

def split_dataset(x, y, test_proportion, random_generator=default_rng(60012)):
    """ Split dataset into training and test sets, according to the given 
        test set proportion.
    
    Args:
        x (np.ndarray): Instances, numpy array with shape (N,K)
        y (np.ndarray): Class labels, numpy array with shape (N,)
        test_proportion (float): the desired proportion of test examples 
                                 (0.0-1.0)
        random_generator (np.random.Generator): A random generator

    Returns:
        tuple: returns a tuple of (x_train, x_test, y_train, y_test) 
               - x_train (np.ndarray): Training instances shape (N_train, K)
               - x_test (np.ndarray): Test instances shape (N_test, K)
               - y_train (np.ndarray): Training labels, shape (N_train, )
               - y_test (np.ndarray): Test labels, shape (N_train, )
    """
    n_test = round(len(x)*test_proportion)
    n_train = len(x) - n_test
    shuffled_indices = random_generator.permutation(len(x))
    
    x_test = x[shuffled_indices[n_train:]]
    y_test = y[shuffled_indices[n_train:]]
    
    x_train = x[shuffled_indices[:n_train]]
    y_train = y[shuffled_indices[:n_train]]
    
    
    return (x_train, x_test, y_train, y_test)

#x, y = read_dataset("wifi_db/clean_dataset.txt")
#x_train, x_test, y_train, y_test = split_dataset(x, y, test_proportion=0.2)

