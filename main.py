from classifier.tree import DecisionTreeClassifier
from classifier.visualiser import visualise
from evaluation.evaluation_metrics import EvaluationMetrics
from evaluation.prune_class import TreePruning
from data_manipulation.load_dataset import load_dataset
from data_manipulation.split_dataset import split_dataset

def print_results(clean = True):
    if clean == True:
        #make initial prediction on test set for clean data
        filepath = 'wifi_db/clean_dataset.txt'
    else:
        filepath = 'wifi_db/noisy_dataset.txt'
    
    X, y = load_dataset(filepath)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_proportion = 0.2)  ## change the split
    tree_clf = DecisionTreeClassifier(max_depth=100)
    tree_clf.fit(X_train, y_train)
    y_test_predicted = tree_clf.predict(X_test)

    #perform evaluation for clean data
    metrics = EvaluationMetrics()
    #confusion_mat = metrics.compute_confusion_matrix()
    #accuracy = metrics.compute_accuracy()
    #precision, recall, f1_score = metrics.compute_precision_recall_f1()

    #perform cross-validation evaluation
    metrics.evaluate_CV(X, y)

    #tree = tree_clf
    #tree_prune = TreePruning(tree, X_train, X_test, y_train, y_train)
    #tree_prune.main_prune()


if __name__ == '__main__':
    print_results()

