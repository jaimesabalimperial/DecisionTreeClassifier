===========================================================
Instructions for running the decision tree algorithm
===========================================================

Contents
-----------------------------------------------------------
1. Classifier folder: 
    a. `Node.py`: Creating the node class, which will be called throughout 
    the code to instance the nodes of the tree
    b. `tree.py`: Decision tree class which includes all the functions to 
    generate the tree. 
2. data_manipulation:
    a. `load_dataset.py`: Function to load the dataset from the txt files. 
    It will return x and y numpy arrays.
    b. `split_dataset.py`: Function to split the dataset into x_train, x_test, y_train, y_test
3. evaluation:
    a. `cross_validation.py`: Cross validation class with functions to perform he k-folds
    b. `evaluation_metrics.py`: Evaluation metrics including 
    accuracy, recall, precision and F1 score
    c. `prune.py`: Tree pruning class
4. wifi_db:
    a. `clean_dataset.py`
    b. `noisy_dataset.py`
5. `main.py`: Main file to run that calls all the functions and classes


How to run the codes
-----------------------------------------------------------
1. Run the `main.py` file 
2. To change whether the clean or noisy data is used, chage the input from the print_results() function
in line 27 to "Clean" or "Noisy" respectively.
This will automatically adjust the path to match the data you require.
3. To choose whether or not to prune the tree, the pruning boolean in line 20 (evaluate_CV() function) 
can be changed to True when pruning or False when not pruning.
- When pruning is set to True, nested cross-validation is performed to evaluate the pruned trees, 
with 10 outer folds and 9 inner folds
- When pruning is set to False, simple cross-validation is performed to evaluate the decision tree classifier.

