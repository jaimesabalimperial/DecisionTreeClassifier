class Node:
    def __init__(self):
        self.feature_num = 0
        self.split_val = 0
        self.left_daughter = None
        self.right_daughter = None
        self.predicted_room = None
        self.is_leaf = False
        self.parent = None
        self.depth = 0
        self.max_depth = 0