import matplotlib.pyplot as plt 
from matplotlib.patches import Patch
import numpy as np
import random


class VisualiseTree():

    def __init__(self, tree_clf, pruning):
        self._tree = tree_clf
        self._nodes = []
        self.pruning = pruning
        self.parse_tree(tree_clf)
        self.parents = {}
    
 
    # traverse tree to get co-ordinates of nodes (and store the parent node - for plotting)
    def parse_tree(self, tree_clf, x_loc=0, depth=0, prev_x_loc=0, prev_depth=0):
        """"""
        self._nodes.append((x_loc, depth, prev_x_loc, prev_depth, tree_clf.feature_num,
                           tree_clf.split_val, tree_clf.is_leaf, tree_clf.predicted_room)
                          )

        if not tree_clf.is_leaf:
            self.parse_tree(tree_clf.left_daughter, x_loc - 2**(depth+3.2), depth-1, x_loc, depth)
            self.parse_tree(tree_clf.right_daughter, x_loc + 2**(depth+3.2), depth-1, x_loc, depth)
    

    def plot(self):
        """"""

        fig = plt.figure() 
        ax = fig.add_subplot(111)

        attributes = ["x_loc", "depth", "prev_x_loc", "prev_depth", "feature_num", 
                      "split_val", "is_leaf", "predicted_room"]

        attribute_dictionary = {}

        for i, att in enumerate(attributes):
            attribute_dictionary[att] = [node[i] for node in self._nodes]

        x_loc = attribute_dictionary["x_loc"]
        depth = attribute_dictionary["depth"]
        prev_x = attribute_dictionary["prev_x_loc"]
        prev_depth = attribute_dictionary["prev_depth"]
        feature_num = attribute_dictionary["feature_num"]
        split_val = attribute_dictionary["split_val"]
        is_leaf = attribute_dictionary["is_leaf"]
        predicted_room = attribute_dictionary["predicted_room"]

        parent_colors = {}

        #loop over all nodes and assign a text box to them 
        for i in range(len(x_loc)):

            #assign same color to line of nodes with same parent
            if (prev_x[i], prev_depth[i]) not in parent_colors:
                r = random.random()
                b = random.random()
                g = random.random()
                color = (r, g, b)

                parent_colors[(prev_x[i], prev_depth[i])] = color
            else:
                color = parent_colors[(prev_x[i], prev_depth[i])]
            
            if is_leaf[i]:
                textstr = '\n'.join(('leaf ',
                                     'prediction = %i' % (int(predicted_room[i])))

                                   )
                ax.text(x_loc[i],depth[i], textstr, fontsize=3.1, weight = "bold", 
                        bbox=dict(facecolor='white', edgecolor='green', lw=0.5))

            elif depth[i] == 0:
                textstr = '\n'.join(('root ',
                                     'split_val = %.1f' % (split_val[i]),
                                     'feature =  %i' % (int(feature_num[i])))
                                    )
                ax.text(x_loc[i],depth[i], textstr, fontsize=3.1, weight = "bold",  
                        bbox=dict(facecolor='white', edgecolor='red', lw=0.5))

            else:
                textstr = '\n'.join((
                                     'split_val =  %.1f' % (split_val[i]),
                                     'feature =  %i' % (int(feature_num[i])))
                                    )

                ax.text(x_loc[i],depth[i],textstr, fontsize=3.1, weight = "bold", 
                        bbox=dict(facecolor='white', edgecolor='blue', lw=0.5))
            
            
            ax.plot([prev_x[i], x_loc[i]], [prev_depth[i],depth[i]], c=color, linewidth=0.65)
        
        legend_elements = [Patch(facecolor='white', edgecolor='r',label='Root Node'), 
                           Patch(facecolor='white', edgecolor='b',label='Normal Node'), 
                           Patch(facecolor='white', edgecolor='g',label='Leaf Node')]

        if self.pruning:
            title_txt = "Sample Pruned Decision Tree Classifier for Outer Fold #0 and Inner Fold #0"
        else:
            title_txt = "Sample Decision Tree Classifier for Fold #0"

        plt.title(title_txt)
        plt.ylabel(("Depth of Tree"))
        plt.legend(handles=legend_elements, loc = "upper left", fontsize=6)
        plt.show()
        

