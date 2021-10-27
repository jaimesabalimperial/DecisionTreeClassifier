from classifier.tree import DecisionTreeClassifier
from classifier.visualiser import visualise
from evaluation.evaluation_metrics import EvaluationMetrics, k_fold_evaluation
from data_manipulation.load_dataset import load_dataset
from data_manipulation.split_dataset import split_dataset


if __name__ == '__main__':
    #make initial prediction on test set for clean data
    filepath = 'wifi_db/clean_dataset.txt'
    X, y = load_dataset(filepath)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_proportion = 0.2)  ## change the split
    tree_clf = DecisionTreeClassifier(max_depth=100)
    tree_clf.fit(X_train, y_train)
    y_test_predicted = tree_clf.predict(X_test)

    #perform evaluation for clean data
    metrics = EvaluationMetrics(y_test, y_test_predicted)
    confusion_mat = metrics.compute_confusion_matrix()
    accuracy = metrics.compute_accuracy()
    precision, recall, f1_score = metrics.compute_precision_recall_f1()
    k_fold_evaluation(X, y)

    #visualise tree
    tree = tree_clf.trained_tree
    #visualise(tree)
