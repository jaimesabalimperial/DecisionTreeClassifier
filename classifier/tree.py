import numpy as np
from classifier.node import Node


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.trained_tree = None

    def calculate_H(self, y):
        """Calculates the entropy of a given dataset (D) through the equation:

        H(D) = - sum_{k=1}^{k=K}(p_k*log2(p_k)) where K is the number of unique classes/labels present in the dataset.
        
        Args: 
            y (np.array): array of labels to the data. 

        Returns: 
            H (float): total entropy of the dataset 
        """
        classes = np.unique(y)
        class_probabilities = [np.count_nonzero(y == room)/len(y) for room in classes]
        H = np.sum([-p*np.log2(p) for p in class_probabilities])
        return H

    def find_split(self, X, y):
        """Finds the best way to split the dataset X with labels y into two daughter nodes by computing
        the information gain of each possible (and relevant) split. 
        
        Args: 
            X (np.array): n-dimensional array containing the data of n features. 
            y (np.array): array of labels to the data array X.

        Returns: 
            best_feature (int): column index of the chosen feature from which to split the data in the decision tree.
            splitting_indices (tuple): 2d tuple containing the sorted indices of the left and right sides of the best split, respectively.
            split_val (float): value chosen to split the data (i.e left_side < split_val < right_side).
        """
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
                if split_idx == 0: #if first index move on since can't split data there
                    continue
                #if the label doesn't change between subsequent samples, don't consider split at current index
                current_label = sorted_labels[split_idx]
                prev_label = sorted_labels[split_idx - 1]
                if current_label == prev_label:
                    continue
                #calculate entropy of both sides for given split index and store information gain in dictionary 
                else: 
                    left_entropy = self.calculate_H(sorted_labels[:split_idx]) #entropy of 'left' side of split
                    right_entropy = self.calculate_H(sorted_labels[split_idx:]) #entropy of 'right' side of split
                    num_samples = X.shape[0]
                    remainder = (split_idx/num_samples)*left_entropy + ((num_samples-split_idx)/num_samples)*right_entropy
                    info_gain = data_entropy - remainder 

                    threshold = (sorted_feature_array[split_idx - 1] + sorted_feature_array[split_idx])/2 #mean value of two consecutive data points within a feature
                    indices_left_node, indices_right_node = tuple(sorted_indices[:split_idx]), tuple(sorted_indices[split_idx:])

                    #add info gain to dictionary with the feature pertaining to the split, the indices of the sorted left and right sides, and the threshold
                    information_gains[(feature_idx, (indices_left_node, indices_right_node), threshold)] = info_gain 

        #retrieve the feature, splitting indices and threshold that has the maximum information gain in whole dictionary
        best_split = max(information_gains, key=information_gains.get)
        best_feature = best_split[0]
        splitting_indices = best_split[1]
        split_val = best_split[2]

        return best_feature, splitting_indices, split_val

    def has_pure_class(self, y):
        """Returns True if labels data only contains one unique class, False otherwise.
        
        Args: y (np.array): array of labels to the data. 
        """
        return np.all(y == y[0])

    def find_predicted_room(self, y):
        """Finds the predicted room from the number of occurences of each class present in the labels data. 
        Args: 
            y (np.array): array of labels to the data. 
        """
        classes = np.unique(y)
        num_samples_in_class = [np.sum(y == sample) for sample in classes]

        return classes[np.argmax(num_samples_in_class)]

    def grow_tree(self, X, y, depth=0):
        """Builds a decision tree classifier from training data X and labels y.
        
        Args: 
            X (np.array): n-dimensional array containing the data of n features. 
            y (np.array): array of labels to the data array X.
            depth (int): current depth of decision tree.

        Returns: 
            node (Node()): tree created from a Node() object whose daughters have been filled recursively. 

        """
        node = Node() #create new node

        # split recursively
        if depth < self.max_depth and not self.has_pure_class(y):
            feature_idx, splitting_indices, threshold = self.find_split(X, y)
            X_left, y_left = X[splitting_indices[0],], y[splitting_indices[0],]
            X_right, y_right = X[splitting_indices[1],], y[splitting_indices[1],]
            node.feature_num = feature_idx
            node.split_val = threshold
            node.left_daughter = self.grow_tree(X_left, y_left, depth + 1)
            node.right_daughter = self.grow_tree(X_right, y_right, depth + 1)
        else: 
            node.is_leaf = True #if max depth or only one class shown in labels, set the current node as a leaf node
            node.predicted_room = self.find_predicted_room(y) #predict a room for leaf node

        return node

    def fit(self, X, y):
        """ Fit the training data to the classifier.
        Args: 
            X (np.array): n-dimensional array containing the data of n features. 
            y (np.array): array of labels to the data array X. 
        """
        self.trained_tree = self.grow_tree(X, y)

    def predict(self, X):
        """Predicts a room for each sample of a test dataset X. 
        Args: 
            X (np.array): n-dimensional array containing the data of n features. 
        
        Returns: 
            predicted_values (list): predicted room to each sample in X.
        """
        predicted_values = []
        for sample in X:
            node = self.trained_tree
            while not node.is_leaf:
                if sample[node.feature_num] < node.split_val:
                    node = node.left_daughter
                else:
                    node = node.right_daughter
            predicted_values.append(node.predicted_room)

        return predicted_values


