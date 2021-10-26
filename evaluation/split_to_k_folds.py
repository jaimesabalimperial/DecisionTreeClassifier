from random import randrange, seed
import numpy as np

seed(60012) #set random seed 

def split_to_k_folds(X, y, folds):
    """"""
    test_set = list()
    label_set = list()
    labels_copy = y.tolist()
    dataset_copy = X.tolist()
    fold_size = int(len(X) / folds)
    for _ in range(folds):
        test_set_fold = list()
        label_set_fold = list()

        while len(test_set_fold) < fold_size:
            index = randrange(len(dataset_copy))
            test_set_fold.append(dataset_copy.pop(index))
            label_set_fold.append(labels_copy.pop(index))

        test_set.append(np.array(test_set_fold))
        label_set.append(np.array(label_set_fold))

    return test_set, label_set


def create_data_k_fold(X, y, folds=10):
    """"""
    test_data, test_labels = split_to_k_folds(X, y, folds)
    train_data = []
    train_labels = []
    for i in range(len(test_data)):
        list_data_copy = test_data.copy()
        list_labels_copy = test_labels.copy()
        del list_data_copy[i]
        del list_labels_copy[i]
        train_data.append(np.array(np.concatenate(list_data_copy)))
        train_labels.append(np.array(np.concatenate(list_labels_copy)))
    return train_data, train_labels, test_data, test_labels


