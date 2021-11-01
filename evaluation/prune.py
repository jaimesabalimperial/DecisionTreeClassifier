import numpy as np
from copy import deepcopy

class TreePruning:
    def __init__(self, trained_tree, x_train, x_test, y_train, y_test, metrics):
        self.trained_tree = trained_tree
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.checked_nodes = []
        self.nodes_to_check = []
        self.node_parent = None
        self.identifier = 1
        self.curr_depth = None
        self.max_depth_before = None
        self.max_depth_after = None
        self.metrics = metrics

    def calculate_max_depth(self, node):
        if node is None:
            return 0
    
        else:
            # Compute the depth of each subtree
            lDepth = self.calculate_max_depth(node.left_daughter)
            rDepth = self.calculate_max_depth(node.right_daughter)
    
            # Use the larger one
            if lDepth > rDepth:
                return lDepth + 1
            else:
                return rDepth + 1
    
    def find_node_to_check(self, node):
        if node.left_daughter is None and node.right_daughter is None:
            return

        if node.depth == (self.curr_depth-2) and node not in self.checked_nodes and node.left_daughter.is_leaf and node.right_daughter.is_leaf:
            self.identifier = 1
            self.node_parent = node

        if node.left_daughter is not None:
            self.find_node_to_check(node.left_daughter)

        if node.right_daughter is not None:
            self.find_node_to_check(node.right_daughter)

    def update_trained_tree(self, tree1, tree2):
        y_predicted_test_pre_pruning = np.array(self.predict_tree(self.x_test, tree=tree1))
        y_predicted_test_post_pruning = np.array(self.predict_tree(self.x_test, tree=tree2))
        self.metrics.y = self.y_test
        self.metrics.y_predicted = y_predicted_test_post_pruning
        tree_f1_post_pruning = self.metrics.compute_averaged_f1()
        #print(f'f1 post pruning: {tree_f1_post_pruning}')
        self.metrics.y_predicted = y_predicted_test_pre_pruning
        tree_f1_pre_pruning = self.metrics.compute_averaged_f1()
        #print(f'f1 pre pruning: {tree_f1_pre_pruning}')

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
        target_node.left_daughter = None
        target_node.right_daughter = None
        target_node.is_leaf = True
        while target_node.depth > 0:
            target_node = target_node.parent
        return target_node 

    def prune_tree(self):
        self.max_depth_before = self.calculate_max_depth(node=self.trained_tree)
        self.curr_depth = self.calculate_max_depth(node=self.trained_tree)
        while self.curr_depth > 0:
            pre_pruned_tree = self.trained_tree
            self.identifier = 0
            self.find_node_to_check(node=pre_pruned_tree)
            self.checked_nodes.append(self.node_parent)
            post_pruned_tree = deepcopy(self.node_parent)
            post_pruned_tree = self.prune_branch(target_node=post_pruned_tree)

            if self.update_trained_tree(tree1=pre_pruned_tree, tree2=post_pruned_tree):
               self.trained_tree = post_pruned_tree
            self.identifier = 0
            self.find_node_to_check(node=self.trained_tree)
            if self.identifier == 0:
                self.curr_depth -= 1
        
        self.max_depth_after = self.calculate_max_depth(node=self.trained_tree)
        return self.max_depth_before, self.max_depth_after, self.trained_tree
        
