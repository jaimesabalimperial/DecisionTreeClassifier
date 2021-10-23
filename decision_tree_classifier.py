import numpy as np

class DecisionTreeClassifier:
    def __init__(self, width = 50):
        self.width = width
        self.x = np.array([])
        self.y = np.array([])

    def calculate_H(self, left_indices, right_indices):
        """"""
        pass #TO DO#

    def find_best_node_candidate_in_a_feature(self, feature_indices, column):
        """"""
        feature_array = self.x[feature_indices, column]
        sorted_feature_array = feature_array.sort()
        sorted_indices = np.argsort(feature_array)
        list_sorted_indices = list(sorted_indices)
        potential_node_indices_list = [] #tuple of index, corrsponding H, left and right indices

        for i in range(len(list_sorted_indices)):
            if i == 0:
                continue

            label_for_i = self.y[list_sorted_indices[i]]
            label_for_i_minus_1 = self.y[list_sorted_indices[i-1]]
            if label_for_i == label_for_i_minus_1:
                continue
            else:
                pivotal_index = list_sorted_indices[i]
                y_left_indices = list_sorted_indices[:i]
                y_right_indices = list_sorted_indices[i:]
                y_left = self.y[y_left_indices]
                x_left = sorted_feature_array[:i]
                y_right = self.y[y_right_indices]
                y_right = sorted_feature_array[i:]
                h_value = self.calculate_H(y_left, y_right)
                potential_node_indices_list.append = (h_value, pivotal_index, y_left_indices, y_right_indices)

        # Choosing the max H for a given feature
        h_index_tuple= max(potential_node_indices_list, key = lambda item: item[0])

        return h_index_tuple


    def split(self, indices_list):
        """"""
        potential_node_index_dict ={}

        for column in range(self.x.axis[1]):
            h_index_tuple = self.find_best_node_candidate_in_a_feature(feature_indices = indices_list, column)
            #Add tuple to a dictionary of features
            potential_node_index_dict[column]=h_index_tuple

        node_H = max(potential_node_index_dict.values(), key = lambda item: item[1])

        for dict in potential_node_index_dict.keys():
            if dict[0] == node_H:
                return y_left_indices, y_right_indices

    def calculate_H(self, labels_distribution):
        pass #TO DO#

    def grow_tree(self, y_left_indices, y_right_indices):
        """"""
        for indices in [y_left_indices, y_right_indices]:
            y_left_indices, y_right_indices = self.split(x, indices_list = indices)

        return y_left_indices, y_right_indices

    def fit(self, x, y):
        """ Fit the training data to the classifier.
        """
        self.x = x
        self.y = y
        inital_indices = list(range(0, x.shape[1]))
        y_left_indices, y_right_indices = self.split(x, indices_list = inital_indices)
        self.depth += 1
        '''100% wrong - need to rethink recursion'''
        while self.depth<100:
            if np.all(y == y[[y_left_indices][0]]):
                return self.depth
            else:
                self.grow_tree(y_left_indices, y_right_indices)
