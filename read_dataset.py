import numpy as np

def read_dataset(filepath):
    """ Read in the dataset from the specified filepath

    Args:
        filepath (str): The filepath to the dataset file

    Returns:
        tuple: returns a tuple of (x, y, classes), each being a numpy array. 
               - x is a numpy array with shape (N, K), 
                   where N is the number of instances
                   K is the number of features/attributes
               - y is a numpy array with shape (N, ), and each element should be 
                   an integer from 0 to C-1 where C is the number of classes 
               - classes : a numpy array with shape (C, ), which contains the 
                   unique class labels corresponding to the integers in y
    """
    x = []
    y = []
    for line in open(filepath):
        if line.strip() != "":  # handle empty rows in file
            row = line.rstrip().split('\t')
            x.append(list(map(float, row[:-1]))) 
            y.append(int(row[-1]))

    x = np.array(x)
    y = np.array(y)
    return (x, y)

#x, y = read_dataset("wifi_db/clean_dataset.txt")
