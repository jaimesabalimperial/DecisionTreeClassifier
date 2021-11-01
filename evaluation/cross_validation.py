import numpy as np
from numpy.random import default_rng


class CrossValidation:

    def __init__(self):
        seed = 60012 #set random seed 
        rg = default_rng(seed)
        self.folds = 10
        self.random_generator = rg

    def k_fold_split(self, x, y):
        """ Split n_instances into n mutually exclusive splits at random.

        Args:
            n_splits (int): Number of splits
            n_instances (int): Number of instances to split
            random_generator (np.random.Generator): A random generator

        Returns:
            list: a list (length n_splits). Each element in the list should contain a 
                numpy array giving the indices of the instances in that split.
        """

        # generate a random permutation of indices from 0 to n_instances
        shuffled_indices = self.random_generator.permutation(len(x))

        # split shuffled indices into almost equal sized splits
        split_indices = np.array_split(shuffled_indices, self.folds)

        return split_indices

    def train_test_k_fold(self, x, y):
        """ Generate train and test indices at each fold.
        
        Args:
            n_folds (int): Number of folds
            n_instances (int): Total number of instances
            random_generator (np.random.Generator): A random generator

        Returns:
            list: a list of length n_folds. Each element in the list is a list (or tuple) 
                with two elements: a numpy array containing the train indices, and another 
                numpy array containing the test indices.
        """

        # split the dataset into k splits
        split_indices = self.k_fold_split(x, y)

        folds = []
        for k in range(self.folds):
            # pick k as test
            test_indices = split_indices[k]

            # combine remaining splits as train
            # this solution is fancy and worked for me
            # feel free to use a more verbose solution that's more readable
            train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])

            folds.append([train_indices, test_indices])

        return folds


    