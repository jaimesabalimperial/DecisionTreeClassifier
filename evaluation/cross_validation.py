from random import seed, sample
import numpy as np

seed(60012) #set random seed 

class CrossValidation:
    def __init__(self, X, y):
        self.folds = 10
        self.testval_folds = int(0.1*self.folds)
        self.validation = False
        self.X = X   
        self.y = y

    def get_random_folded_data(self, random_fold_indices):
        """"""
        test_indices_list = [random_fold_indices[i] for i in range(self.testval_folds)]
        test_data = []
        test_labels = []
        for fold in test_indices_list:
            test_data.append(self.X[fold])
            test_labels.append(self.y[fold])

        trainval_indices_list = [random_fold_indices[i] for i in range(self.testval_folds, self.folds)]
        trainval_data = []
        trainval_labels = []
        for fold in trainval_indices_list:
            trainval_data.append(self.X[fold])
            trainval_labels.append(self.y[fold])

        #first split of test data and labels
        test_data = np.concatenate(test_data)
        test_labels  = np.concatenate(test_labels)

        trainval_data = np.array(trainval_data)
        trainval_labels = np.array(trainval_labels)

        return test_data, test_labels, trainval_data, trainval_labels

    def split_to_k_folds(self):
        """"""
        random_fold_indices = []
        fold_size = int(len(self.X) / self.folds)

        for _ in range(self.folds):
            if len(random_fold_indices) != 0:
                current_chosen_indices = [idx for fold in random_fold_indices for idx in fold]
                available_indices = [i for i in range(len(self.X)) if i not in current_chosen_indices]
            else: 
                available_indices = [i for i in range(len(self.X))]
            
            random_fold_indices.append(sample(available_indices, fold_size)) #get random indices within range of data for each fold
        

        if self.validation:
            test_data, test_labels, trainval_data, trainval_labels = self.get_random_folded_data(random_fold_indices)
            train_data, train_labels, val_data, val_labels = self.iterated_folds(trainval_data, trainval_labels)

            return train_data, train_labels, val_data, val_labels, test_data, test_labels

        else: 
            k_folded_data = [self.X[fold] for fold in random_fold_indices]
            k_folded_labels = [self.y[fold] for fold in random_fold_indices]

            return k_folded_data, k_folded_labels

    def iterated_folds(self, folded_data, folded_labels):
        """"""
        majority_data, majority_labels = [], []
        minority_data, minority_labels = folded_data, folded_labels

        if self.validation:
            num_iterations = int(self.folds - self.testval_folds)
        else: 
            num_iterations = self.folds

        for i in range(num_iterations):
            folded_data_copy = folded_data.copy()
            folded_labels_copy = folded_labels.copy()
            del folded_data_copy[i]
            del folded_labels_copy[i]
            majority_data.append(np.array(np.concatenate(folded_data_copy)))
            majority_labels.append(np.array(np.concatenate(folded_labels_copy)))

        return majority_data, majority_labels, minority_data, minority_labels

    def generate_CV_data(self):
        """"""
        self.validation = False 

        k_folded_data, k_folded_labels = self.split_to_k_folds()

        train_data, train_labels, test_data, test_labels = self.iterated_folds(k_folded_data, k_folded_labels)

        return train_data, train_labels, test_data, test_labels

    def nested_CV_data(self):
        """"""
        folded_data, folded_labels = self.split_to_k_folds()
        folded_data_copy = folded_data.copy()
        folded_labels_copy = folded_labels.copy()

        #10/10/80 (test/val/train)
        #have 1 fold worth of test data, and 9 folds of validation + 8*9 folds of training data
        self.validation = True

        test_data, test_labels = [], [], 
        train_data, train_labels, val_data, val_labels = {}, {}, {}, {}

        for i in range(self.folds):
            test_data.append(folded_data[i])
            test_labels.append(folded_labels[i])

            folded_data_copy = [folded_data[i] for j in range(self.folds) if j != i]
            folded_labels_copy = [folded_labels[i] for j in range(self.folds) if j != i]

            train_data_i, train_labels_i, val_data_i, val_labels_i = self.iterated_folds(folded_data_copy, folded_labels_copy)

            train_data[i] = train_data_i
            train_labels[i] = train_labels_i    
            val_data[i] = val_data_i
            val_labels[i] = val_labels_i

        return train_data, train_labels, val_data, val_labels, test_data, test_labels