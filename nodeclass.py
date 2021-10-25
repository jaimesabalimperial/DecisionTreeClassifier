class Node:
    def __init__(self, feature_num, split_val, left_node, right_node): #add leaf
        self.feature_num = feature_num
        self.split_val = split_val
        self.left_node = left_node
        self.right_node = right_node