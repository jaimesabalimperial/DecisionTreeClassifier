from evaluation.evaluation_metrics import EvaluationMetrics
from classifier.node import Node
import numpy as np

class TreePruning1:
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




class TreePruning2:
    def __init__(self, trained_tree, x_train, x_test, y_train, y_test):
        self.trained_tree = trained_tree
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.checked_nodes = []

    
    def query_update_tree(self,node_to_find, node):

        if node is None:
            return    
        else:
            pass

        if node.left_daughter is not None:
            self.query_update_tree(node_to_find, node.left_daughter)

        if (node.depth == node_to_find.depth) and (node.feature_num == node_to_find.feature_num) and (node.split_val == node_to_find.split_val):
            node.left_daughter = None
            node.right_daughter = None
            node.is_leaf = True
            return node

        if node.right_daughter is not None:
            self.query_update_tree(node_to_find, node.right_daughter)

    def prune_left_most_parent(self, node):
        ''' Find next deepest, left most node for pruning. Track trained tree in parallel to 
            potentially update its node
        '''
        if node is None:
            return

        if node.left_daughter.is_leaf == False and node.left_daughter not in self.checked_nodes:
            self.prune_left_most_parent(node.left_daughter)
        
        if node.left_daughter.is_leaf == True and node.left_daughter not in self.checked_nodes:
            if node.right_daughter.is_leaf == True and node not in self.checked_nodes:
                # Add left and right data
                node_to_find = Node()
                node_to_find.depth = node.depth 
                node_to_find.feature_num = node.feature_num
                node_to_find.split_val = node.split_val
                node.left_daughter = None
                node.right_daughter = None
                node.is_leaf = True
                self.checked_nodes.append(node)
            else:
                self.prune_left_most_parent(node.right_daughter)
        else:
            return None

        return node

    def compute_accuracy(self, y, y_predicted):
        """
        """
        assert len(y) == len(y_predicted)
        try:
            return np.sum(y == y_predicted) / len(y)
        except ZeroDivisionError:
            return 0
         

    def update_trained_tree(self, tree1, tree2):
        """
        """
        #metrics = EvaluationMetrics() 

        y_predicted_test_pre_pruning = np.array(self.predict_tree(self.x_test, tree = tree1))
        y_predicted_test_post_pruning = np.array(self.predict_tree(self.x_test, tree = tree2))
        tree_accuracy_post_pruning= self.compute_accuracy(self.y_test, y_predicted_test_post_pruning)
        tree_accuracy_pre_pruning = self.compute_accuracy(self.y_test,y_predicted_test_pre_pruning) 

        return  tree_accuracy_post_pruning >= tree_accuracy_pre_pruning

    def predict_tree(self, X, tree):
        """
        """
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
        

    def prune_tree(self):
        """"""
        #pre_pruned_tree = self.trained_tree
        #node = pre_pruned_tree
        #node_is_parent = (node.left_daughter.is_leaf == True) and (node.right_daughter.is_leaf == True)
        pre_pruned_tree = self.trained_tree
        if self.prune_left_most_parent(pre_pruned_tree) == None:
            print(pre_pruned_tree)
            y_predicted_test_pre_pruning = np.array(self.predict_tree(self.x_test, tree = pre_pruned_tree))
            print(self.compute_accuracy(self.y_test, y_predicted_test_pre_pruning))
            return pre_pruned_tree
        else:
            post_pruned_tree = self.prune_left_most_parent(pre_pruned_tree)

            #update left and right daughters to none and isleaf.yes
            #do not know how to concatenate node and path (node+path).isleaf = True
            if self.update_trained_tree(tree1 = pre_pruned_tree, tree2 = post_pruned_tree):
                self.query_update_tree(node_to_find=post_pruned_tree, node=pre_pruned_tree)  

            self.trained_tree = post_pruned_tree
            self.prune_tree()