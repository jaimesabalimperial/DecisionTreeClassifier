import matplotlib.pyplot as plt 
import numpy as np

class VisualiseTree:

    def __init__(self):
        self.figheight = 20
        self.figwidth = 60
        fig = plt.figure(figsize=(self.figwidth, self.figheight)) #make figure
        ax = fig.gca()

        plt.title("Decision Tree Classifier")

        self.fig = fig
        self.ax = ax
        self.line_yrange = None
        self.line_xrange = None
        self.parent_box_height = None
        self.parent_box_width = None
        self.tree = None
        self.r = None
        self.lines = []
        self.node_boxes = []
        self.leaf_boxes = []

    def get_box_dim(self, box):
        """"""
        r = self.fig.canvas.get_renderer()

        t = self.ax.text(box[0], box[1], box[2])

        bb = t.get_window_extent(renderer=r).inverse_transformed(self.ax.transData)
        width = bb.width
        height = bb.height

        return width, height

    def place_text(self, depth, split_val, feature_num, parent_loc):
        if depth == 0:
            textstr = '\n'.join(('root',
                                 'split_val: %.1f' % (split_val),
                                 'feature_num: %i' % (feature_num))
                               )

            x = parent_loc[0]
            y = parent_loc[1]

            text_box = (x, y, textstr)
            
            self.node_boxes.append(text_box)

        else: 
            textstr = '\n'.join(('depth: %i' % (depth),
                                 'split_val: %.1f' % (split_val),
                                 'feature_num: %i' % (feature_num))
                               )

            x = self.line_xrange[0]
            y = self.line_yrange[0] - self.parent_box_height/2

            text_box = (x, y, textstr)

            self.node_boxes.append(text_box)

    
    def make_tree_lines(self, parent_loc):
        """"""
        self.line_xrange = (parent_loc[0] - self.figwidth*self.r, parent_loc[0])
        y_max = parent_loc[1] - self.parent_box_height/2
        self.line_yrange = (parent_loc[1]*(1 - 1/self.tree.max_depth), y_max)

        x_left_connection = self.line_xrange
        x_right_connection = (self.line_xrange[1], -self.line_xrange[0])

        #lines connecting parent to daughters
        line1 = (x_left_connection, self.line_yrange)
        line2 = (x_right_connection, self.line_yrange)

        self.lines.append((line1, line2))

    def create_tree_objects(self, tree_clf, depth = 0, parent_loc = (0, 19)):
        """Visualises the decision tree classifier. """
        self.tree = tree_clf
        self.r = 2**(depth+1)

        #recursively go through tree
        if not tree_clf.is_leaf: 
            depth = tree_clf.depth
            split = tree_clf.split_val
            feature = tree_clf.feature_num

            self.place_text(depth, split, feature, parent_loc)
            text_box = self.node_boxes[depth]
            self.parent_box_width, self.parent_box_height = self.get_box_dim(text_box)

            self.make_tree_lines(parent_loc)
            left_daughter_loc = (self.line_xrange[0], self.line_yrange[0] - self.parent_box_height/2)
            right_daughter_loc = (-self.line_xrange[0], self.line_yrange[0] - self.parent_box_height/2)

            #recursively visualise left and right daughters
            self.create_tree_objects(tree_clf.left_daughter, depth = depth + 1, parent_loc=left_daughter_loc)
            self.create_tree_objects(tree_clf.right_daughter, depth = depth + 1, parent_loc=right_daughter_loc)

        else: 
            predicted_room = tree_clf.predicted_room

            textstr = '\n'.join(('leaf',
                                 'prediction: %i' % (predicted_room))
                                )

            x = self.line_xrange[0]
            y = self.line_yrange[0] - self.parent_box_height/2

            text_box = (x, y, textstr)

            self.leaf_boxes.append(text_box)

    
    def visualise(self, tree_clf):

        self.create_tree_objects(tree_clf)

        for box, (line1, line2) in zip(self.node_boxes, self.lines):

            #plot each nodes' box with the connecting lines coming out of it
            self.ax.text(box[0], box[1], box[2], bbox=dict(facecolor='none', edgecolor='blue'))
            self.ax.plot(line1[0], line1[1])
            self.ax.plot(line2[0], line2[1])

        for leaf_box in self.leaf_boxes:
            self.ax.text(leaf_box[0], leaf_box[1], leaf_box[2], bbox=dict(facecolor='none', edgecolor='green'))


        plt.show()