class Node:
    def __init__(self):
        self.feature_num = 0
        self.split_val = 0
        self.left_daughter = None
        self.right_daughter = None
        self.predicted_room = None
        self.leaf = False