from evaluation.evaluation_metrics import EvaluationMetrics
import numpy as np

class TreePruning:
    def __init__(self, trained_tree, x_train, x_test, y_train, y_test):
        self.trained_tree = trained_tree
        self.try_prune_tree = trained_tree
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.leaf_list = []
        self.sorted_leaf = []
        self.y_predicted_pre_pruning = None
        self.y_predicted_post_pruning = None
        self.metrics = EvaluationMetrics(self.y_test, self.y_predicted_pre_pruning)

    def find_node(self, node, target_node):
        if node == None:
            return

        if node.left_daughter is not None:
            self.find_node(node.left_daughter, target_node)

        if node == target_node:
            node.left_daughter = None
            node.right_daughter = None
            node.is_leaf = True
            return node

        if node.right_daughter is not None:
            self.find_node(node.right_daughter, target_node)

    def get_leaf_list(self, node):
        if node == None:
            return   

        if not node.left_daughter.is_leaf:
            self.get_leaf_list(node.left_daughter)

        if node.left_daughter.is_leaf and node.right_daughter.is_leaf:
            self.leaf_list.append(node)    

        if not node.right_daughter.is_leaf:
            self.get_leaf_list(node.right_daughter)
            
    def sort_depth(self):
        depth_list = []
        for node in self.leaf_list:
            depth_list.append(node.depth)
        
        depth_list = np.argsort(np.array(depth_list))[::-1]
        self.sorted_leaf = [self.leaf_list[i] for i in depth_list]

    def try_prune(self):
        pre_pruned_tree = self.try_prune_tree
        target_node = self.sorted_leaf[0]
        post_pruned_tree = self.find_node(pre_pruned_tree, target_node)

        return (self.update_trained_tree(pre_pruned_tree, post_pruned_tree), post_pruned_tree)

    
    def update_trained_tree(self, tree1, tree2):
        """
        """
        y_predicted_test_pre_pruning = self.predict_tree(self.x_test, tree = tree1)
        y_predicted_test_post_pruning = self.predict_tree(self.x_test, tree = tree2)
        self.y_predicted_pre_pruning = y_predicted_test_pre_pruning
        self.y_predicted_post_pruning = y_predicted_test_post_pruning
        tree_accuracy_post_pruning= self.metrics.compute_accuracy(self.y_test, self.y_predicted_test_post_pruning)
        tree_accuracy_pre_pruning = self.metrics.compute_accuracy(self.y_test, self.y_predicted_test_pre_pruning) 

        return  tree_accuracy_post_pruning >= tree_accuracy_pre_pruning

    def predict_tree(self, X, tree):
        """
        """
        predicted_values = []
        for sample in X:
            node = tree
            print(node)
            print(node.is_leaf)
            while not node.is_leaf:
                if sample[node.feature_num] < node.split_val:
                    node = node.left_daughter
                else:
                    node = node.right_daughter
            predicted_values.append(node.predicted_room)

        return predicted_values
            
    def prune(self):
        
        if len(self.sorted_leaf) != 0:
            if self.try_prune()[0]:
                self.trained_tree = self.try_prune()[1]
                if self.sorted_leaf[0].parent.is_leaf:
                    self.sorted_leaf.append(self.sorted_leaf[0].parent)
                self.sorted_leaf.remove(self.sorted_leaf[0])
                self.leaf_list = self.sorted_leaf
                self.sort_depth()
            else:
                self.try_prune_tree = self.trained_tree
                self.sorted_leaf.remove(self.sorted_leaf[0])
                self.leaf_list = self.sorted_leaf
        
        if len(self.sorted_leaf) == 0:
            return
        else:
            self.prune()

    def main_prune(self):
        self.get_leaf_list(self.trained_tree)
        self.sort_depth()
        self.prune()
        return self.trained_tree