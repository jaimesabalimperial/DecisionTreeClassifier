import numpy as np
from numpy.core.fromnumeric import argsort
from classifier.tree import DecisionTreeClassifier
#from classifier.visualiser import visualise
#from evaluation.evaluation_metrics import EvaluationMetrics, k_fold_evaluation
from evaluation.evaluation_metrics import EvaluationMetrics
from data_manipulation.load_dataset import load_dataset
from data_manipulation.split_dataset import split_dataset
from classifier.node import Node

from copy import deepcopy

class TreePruning:
    def __init__(self, trained_tree, x_train, x_test, y_train, y_test):
        self.trained_tree = trained_tree
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.nodes_to_check = []
        self.node_parent = None
        self.identifier = 1
        self.max_depth = None

    def find_initial_leaves(self, node):
        if node is None:
            return

        if node.is_leaf:
            self.nodes_to_check.append(node.parent)
            print(node.parent.depth)

        if node.left_daughter is not None:
            self.find_initial_leaves(node.left_daughter)

        if node.right_daughter is not None:
            self.find_initial_leaves(node.right_daughter)

    def update_trained_tree(self, tree1, tree2):
        y_predicted_test_pre_pruning = np.array(self.predict_tree(self.x_test, tree=tree1))
        y_predicted_test_post_pruning = np.array(self.predict_tree(self.x_test, tree=tree2))

        evaluation = EvaluationMetrics()
        evaluation.y = self.y_test
        evaluation.y_predicted = y_predicted_test_post_pruning
        tree_f1_post_pruning = evaluation.compute_averaged_f1()
        print(f'f1 post pruning: {tree_f1_post_pruning}')
        evaluation.y_predicted = y_predicted_test_pre_pruning
        tree_f1_pre_pruning = evaluation.compute_averaged_f1()
        print(f'f1 pre pruning: {tree_f1_pre_pruning}')

        return tree_f1_post_pruning >= tree_f1_pre_pruning

    def predict_tree(self, X, tree):
        predicted_values = []
        for sample in X:
            node = tree
            while not node.is_leaf:
                if sample[node.feature_num] < node.split_val:
                    node = node.left_daughter
                else:
                    node = node.right_daughter
            predicted_values.append(node.predicted_room)

        return predicted_values

    def prune_branch(self, target_node):
        post_pruned_tree = deepcopy(target_node)
        post_pruned_tree.left_daughter = None
        post_pruned_tree.right_daughter = None
        post_pruned_tree.is_leaf = True
        while post_pruned_tree.depth > 0:
            post_pruned_tree = post_pruned_tree.parent
        return post_pruned_tree

    def prune_tree(self):
        self.find_initial_leaves(self.trained_tree)
        self.nodes_to_check = list(set(self.nodes_to_check))
        while self.nodes_to_check:
            pre_pruned_tree = self.trained_tree
            self.node_parent = self.nodes_to_check[0]
            post_pruned_tree = self.prune_branch(target_node=self.node_parent)
            if self.update_trained_tree(tree1=pre_pruned_tree, tree2=post_pruned_tree):
                self.trained_tree = post_pruned_tree
                self.nodes_to_check.remove(self.node_parent)
                if self.node_parent.left_daughter.is_leaf and self.node_parent.right_daughter.is_leaf:
                    self.node_parent = self.node_parent.parent
                    self.nodes_to_check.append(self.node_parent)
            else:
                self.nodes_to_check.remove(self.node_parent)

if __name__ == '__main__':
    #make initial prediction on test set for clean data
    filepath = 'wifi_db/noisy_dataset.txt'
    #filepath = 'wifi_db/trial_dataset.txt'
    X, y = load_dataset(filepath, clean=False)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_proportion=0.2)  ## change the split
    tree_clf = DecisionTreeClassifier(max_depth=100)
    tree_clf.fit(X_train, y_train)
    y_test_predicted = tree_clf.predict(X_test)
    tree = tree_clf.trained_tree

    # prune tree
    tree_prune = TreePruning(tree, X_train, X_test, y_train, y_test)
    post_pruned = tree_prune.prune_tree()


