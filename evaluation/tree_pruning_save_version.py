

def compute_accuracy(y, y_predicted):
    """Computes the accuracy of the decision tree classifier.
    
    Args: 
        y (np.array): array of labels to the data. 
        y_predicted (list): predicted values only from features data using the built decision tree classifier. 
    """
    assert len(y) == len(y_predicted)
    try:
        return np.sum(y == y_predicted)/len(y)
    except ZeroDivisionError:
        return 0



class Tree_pruning:
    def __init__(self, trained_tree):
        self.trained_tree = trained_tree
    

    def prune_left_most_parent(self, node, path=""):
        ''' Find next deepest, left most node for pruning. Track trained tree in parallel to potentially update its node'''
        if node.left_daughter.is_leaf == True:
            if node.right_daughter.is_leaf == True:
                node.left_daughter = None
                node.right_daughter = None
                node.is_leaf = True
            else:
                updated_path = path+".right_daughter"
                prune_left_most_parent(node.right_daughter, path = updated_path)
        else:
            updated_path = path+".left_daughter"
            prune_left_most_parent(node.left_daughter)
        return node
         

    def update_trained_tree(self, tree1, tree2):
        y_predicted_test_pre_pruning = self.predict_tree(X, tree = tree1)
        y_predicted_test_post_pruning = self.predict_tree(X, tree = tree2)
        tree_accuracy_post_pruning= compute_accuracy(y_test, y_predicted_test_post_pruning)
        tree_accuracy_pre_prepruning = compute_accuracy(y_test,y_predicted_test_pre_pruning)
        return  tree_accuracy_post_incremental_pruning > tree_accuracy_pre_incremental_pruning:


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
        
    node = self.trained_tree

    def prune_tree(self, pre_pruned_tree = self.trained_tree):
        node_is_parent = (node.left_daughter.is_leaf == True) and (node.right_daughter.is_leaf == True)
        path, post_pruned_tree = self.prune_left_most_parent(pre_pruned_tree)
            if self.update_trained_tree(tree1 = pre_pruned_tree, tree2 = post_pruned_tree):
                #update left and right daughters to none and isleaf.yes
                # do not know how to concatenate node and path (node+path).isleaf = True
        while post_pruned_tree is not node_is_parent:
            pre_pruned_tree = post_pruned_tree
            self.prune_tree(pre_pruned_tree=post_pruned_tree)
 

        
 



