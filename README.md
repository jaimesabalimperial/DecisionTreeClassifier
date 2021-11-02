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


How to run the code
-----------------------------------------------------------
1. There are 4 possibilities to run the code depending on the user needs, see the different commands below:
    a. Generating a decision tree using the clean data without pruning: 
    `python3 main.py clean` (clean can be anything from ["clean", "Clean", "CLEAN", "c", "C"])
    b. Generating a decision tree using the noisy data without pruning: 
    `python3 main.py noisy` (noisy can be anything from ["noisy", "Noisy", "NOISY", "n", "N"])
    c. Generating a decision tree and pruning it using the clean data:
    `python3 main.py clean prune` (clean can be anything from ["clean", "Clean", "CLEAN", "c", "C"], 
    prune can be anything from ["prune", "pruning", "Prune", "p", "P"])
    c. Generating a decision tree and pruning it using the noisy data:
    `python3 main.py noisy prune` (noisy can be anything from ["noisy", "Noisy", "NOISY", "n", "N"], 
    prune can be anything from ["prune", "pruning", "Prune", "p", "P"])

- When pruning the tree, nested cross-validation is performed to evaluate the pruned trees, 
with 10 outer folds and 9 inner folds
- When not pruning tree, simple cross-validation is performed to evaluate the decision tree classifier.

