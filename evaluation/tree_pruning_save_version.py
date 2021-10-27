from evaluation.evaluation_metrics import EvaluationMetrics
from data_manipulation.load_dataset import load_dataset
from data_manipulation.split_dataset import split_dataset

class TreePruning:
    def __init__(self, trained_tree):
        #make initial prediction on test set for clean data
        filepath = 'wifi_db/clean_dataset.txt'
        X, y = load_dataset(filepath)
        x_train, x_test, y_train, y_test = split_dataset(X, y, test_proportion = 0.2)  ## change the split
        
        self.trained_tree = trained_tree
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def prune_left_most_parent(self, node, path=""):
        ''' Find next deepest, left most node for pruning. Track trained tree in parallel to 
            potentially update its node
        '''
        if node.left_daughter.is_leaf == True:
            if node.right_daughter.is_leaf == True:
                node.left_daughter = None
                node.right_daughter = None
                node.is_leaf = True
            else:
                updated_path = path + ".right_daughter"
                self.prune_left_most_parent(node.right_daughter, path = updated_path)
        else:
            updated_path = path + ".left_daughter"
            self.prune_left_most_parent(node.left_daughter)

        return node
         

    def update_trained_tree(self, tree1, tree2):
        """
        """
        metrics = EvaluationMetrics() 

        y_predicted_test_pre_pruning = self.predict_tree(self.x_test, tree = tree1)
        y_predicted_test_post_pruning = self.predict_tree(self.x_test, tree = tree2)
        tree_accuracy_post_pruning= metrics.compute_accuracy(self.y_test, y_predicted_test_post_pruning)
        tree_accuracy_pre_pruning = metrics.compute_accuracy(self.y_test,y_predicted_test_pre_pruning) 

        return  tree_accuracy_post_pruning > tree_accuracy_pre_pruning

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
        pre_pruned_tree = self.trained_tree
        node = pre_pruned_tree
        node_is_parent = (node.left_daughter.is_leaf == True) and (node.right_daughter.is_leaf == True)
        path, post_pruned_tree = self.prune_left_most_parent(pre_pruned_tree)

        #update left and right daughters to none and isleaf.yes
        #do not know how to concatenate node and path (node+path).isleaf = True
        if self.update_trained_tree(tree1 = pre_pruned_tree, tree2 = post_pruned_tree):
            pass
        while post_pruned_tree is not node_is_parent:
            pre_pruned_tree = post_pruned_tree
            self.prune_tree(pre_pruned_tree=post_pruned_tree)
 

        
 



