from classifier.tree import DecisionTreeClassifier, compute_accuracy
from data_manipulation.read_dataset import read_dataset
from data_manipulation.split_dataset import split_dataset


if __name__ == '__main__':
    filepath = 'wifi_db/clean_dataset.txt'
    X, y = read_dataset(filepath)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_proportion = 0.2)  ## change the split
    tree_clf = DecisionTreeClassifier(max_depth=100)
    tree_clf.fit(X_train, y_train)
    y_predicted_train = tree_clf.predict(X_train)
    y_predicted_test = tree_clf.predict(X_test)

    print(f"The training accuracy is: {compute_accuracy(y_train, y_predicted_train)}")
    print(f"The test accuracy is: {compute_accuracy(y_test, y_predicted_test)}")