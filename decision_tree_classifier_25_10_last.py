import numpy as np
from read_dataset import read_dataset
from split_dataset import split_dataset

class Node:
    def __init__(self):
        self.feature_num = 0
        self.split_val = 0
        self.left_node = None
        self.right_node = None
        self.predicted_room = None

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def calculate_H(self, y):
        classes = np.unique(y)
        class_probabilities = [np.count_nonzero(y == room)/len(y) for room in classes]
        H = np.sum([-p*np.log2(p) for p in class_probabilities])
        return H

    def find_split(self, X, y):
        information_gains = {}
        data_entropy = self.calculate_H(y)

        for feature_idx in range(X.shape[1]):
            # sort features
            feature_array = X[:, feature_idx]
            sorted_feature_array = sorted(feature_array)
            sorted_indices = np.argsort(feature_array)

            # sort labels
            sorted_labels = y[sorted_indices]

            # decide the best split
            for split_idx in range(len(sorted_feature_array)):
                if split_idx == 0:
                    continue
                left_entropy = self.calculate_H(sorted_labels[:split_idx])
                right_entropy = self.calculate_H(sorted_labels[split_idx:])
                num_feature = X.shape[0]
                remainder = (split_idx/num_feature)*left_entropy + ((num_feature-split_idx)/num_feature)*right_entropy
                info_gain = data_entropy - remainder
                threshold = (sorted_feature_array[split_idx - 1] + sorted_feature_array[split_idx])/2
                indices_left_node = sorted_indices[:split_idx]
                indices_right_node = sorted_indices[split_idx:]
                information_gains[(feature_idx, tuple(indices_left_node), tuple(indices_right_node), threshold)] = info_gain

        best_split = max(information_gains, key=information_gains.get)
        return best_split[0], best_split[1], best_split[2], best_split[3]

    def has_pure_class(self, y):
        return np.all(y == y[0])

    def find_predicted_room(self, y):
        classes = np.unique(y)
        num_samples_in_class = []
        for sample in classes:
            num_samples_in_class.append(np.sum(y == sample))
        return classes[np.argmax(num_samples_in_class)]

    def grow_tree(self, X, y, depth=0):
        node = Node()
        node.predicted_room = self.find_predicted_room(y)
        # split recursively
        if depth < self.max_depth and not self.has_pure_class(y):
            feature_idx, indices_left_node, indices_right_node, threshold = self.find_split(X, y)
            X_left, y_left = X[indices_left_node, :], y[indices_left_node, ]
            X_right, y_right = X[indices_right_node, :], y[indices_right_node, ]
            node.feature_num = feature_idx
            node.split_val = threshold
            node.left_node = self.grow_tree(X_left, y_left, depth + 1)
            node.right_node = self.grow_tree(X_right, y_right, depth + 1)
        return node

    def fit(self, X, y):
        """ Fit the training data to the classifier.
        """
        self.trained_tree = self.grow_tree(X, y)

    def predict(self, X):
        predicted_values = []
        for sample in X:
            node = self.trained_tree
            while node.left_node is not None:
                if sample[node.feature_num] < node.split_val:
                    node = node.left_node
                else:
                    node = node.right_node
            predicted_values.append(node.predicted_room)
        return predicted_values

def compute_accuracy(y, y_predicted):
    assert len(y) == len(y_predicted)
    try:
        return np.sum(y == y_predicted)/len(y)
    except ZeroDivisionError:
        return 0


filepath = 'wifi_db/clean_dataset.txt'
X, y = read_dataset(filepath)
X_train, X_test, y_train, y_test = split_dataset(X, y, test_proportion=0.2)  ## change the split
tree_clf = DecisionTreeClassifier(max_depth=100)
tree_clf.fit(X_train, y_train)
y_train_predicted = tree_clf.predict(X_train)
y_test_predicted = tree_clf.predict(X_test)
print(f"The training accuracy is: {compute_accuracy(y_train, y_train_predicted)}")
print(f"The test accuracy is: {compute_accuracy(y_test, y_test_predicted)}")