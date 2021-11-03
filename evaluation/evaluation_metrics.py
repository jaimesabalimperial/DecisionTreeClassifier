import numpy as np
from classifier.visualiser import VisualiseTree
from evaluation.cross_validation import CrossValidation
from classifier.tree import DecisionTreeClassifier
from evaluation.prune import TreePruning


class EvaluationMetrics:
    def __init__(self):
        self.y = None
        self.y_predicted = None

    def get_confusion_matrix(self):
        """
        """
        # compute TP
        num_of_classes = len(list(set(self.y)))
        confusion_mat = np.empty([num_of_classes, num_of_classes])

        for row, true_class in enumerate(list(set(self.y))):
            for col, pred_class in enumerate(list(set(self.y))):
                confusion_mat[row][col] = sum((self.y == true_class) & (self.y_predicted == pred_class))

        return confusion_mat

    def get_accuracy(self):
        """
        """
        assert len(self.y) == len(self.y_predicted)

        try:
            return np.sum(self.y == self.y_predicted) / len(self.y)

        except ZeroDivisionError:
            return 0

    def get_precision_recall_f1(self):
        """
        """
        #get confusion matrix for data labels and predictions
        confusion_matrix = self.get_confusion_matrix()
        metrics = []

        #find precision, recall, and labels for all classes
        for room in range(len(list(set(self.y)))):

            #all cases predicted to be negative, want to maintain information by inputting np.nan
            try: 
                precision = confusion_matrix[room][room] / np.array(confusion_matrix).sum(axis=1)[room]
            except ZeroDivisionError:
                precision = np.nan

            #no positives in input data, want to maintain information by inputting np.nan
            try: 
                recall = confusion_matrix[room][room] / np.array(confusion_matrix).sum(axis=0)[room]
            except ZeroDivisionError:
                recall = np.nan

            #f1_score is only nan when model had no opportunity to identify true positives, 
            #but it had no false negatives, else it is zero
            try: 
                curr_f1 = (2 * precision * recall) / (precision + recall)

            except ZeroDivisionError:
                if recall == np.nan and precision == np.nan: 
                    curr_f1 = np.nan
                else: 
                    curr_f1 = 0
            
            metrics.append((precision, recall, curr_f1))

        return metrics

    def compute_averaged_f1(self):

        f1_score = list(zip(*self.get_precision_recall_f1()))[2]

        return np.nanmean(f1_score)


    def average_metrics(self, metrics, pruning):
        """
        """
        #get average of all folds
        total_confusion_mat = sum(metrics["Confusion Matrices"])
        sum_of_rows = total_confusion_mat.sum(axis=1)
        normalized_confusion_matrix = total_confusion_mat / sum_of_rows[:, np.newaxis]

        average_accuracy = np.mean(metrics["Accuracies"])
        average_precision = list(map(np.nanmean, zip(*metrics["Precisions"])))
        average_recall = list(map(np.nanmean, zip(*metrics["Recalls"])))
        average_f1_score = list(map(np.nanmean, zip(*metrics["F1 Scores"])))

        #if we are pruning, also calculate average maximum depths before and after 
        if pruning: 
            average_max_depth_before = np.mean(metrics["Max Depth Before"])
            average_max_depth_after = np.mean(metrics["Max Depth After"])

            return (normalized_confusion_matrix, average_accuracy, average_precision, 
                    average_recall , average_f1_score, average_max_depth_before, average_max_depth_after)
        
        else: 
            return (normalized_confusion_matrix, average_accuracy, average_precision, 
                    average_recall , average_f1_score)

    def print_metrics(self, final_average_metrics, pruning):
        """"""
        # print final metrics
        print(f"\nThe total normalised confusion matrix is: \n {final_average_metrics[0]}")
        print(f"\nThe average accuracy is: \n{final_average_metrics[1]}")
        print(f"\nThe average precision for each class is: \n{final_average_metrics[2]}")
        print(f"\nThe average recall for each class is: \n{final_average_metrics[3]}")
        print(f"\nThe average f1 score for each class is: \n{final_average_metrics[4]}")

        if pruning:
            print(f"\nThe average maximum depth before pruning is : \n {final_average_metrics[5]}")
            print(f"\nThe average maximum depth after pruning is : \n {final_average_metrics[6]} \n")
        
        print("\n")

    def evaluate_CV(self, x, y, pruning):
        """
        """
        cv = CrossValidation() #retrieve object to perform cross-validation 

        #simple cross-validation to evaluate decision tree classifier
        if not pruning:
            #initialise metric lists
            metrics_dict = {"Confusion Matrices": [], "Accuracies": [], "Precisions": [], "Recalls": [], "F1 Scores": []}

            folds = 10 
            cv.folds = folds

            print("Currently evaluating: ")
            for i, (train_indices, test_indices) in enumerate(cv.train_test_k_fold(x, y)):
                print("\nFold #", i)
                # get the dataset from the correct splits
                x_train = x[train_indices, :]
                y_train = y[train_indices]
                x_test = x[test_indices, :]
                y_test = y[test_indices]

                tree_clf = DecisionTreeClassifier(max_depth=100)
                tree_clf.fit(x_train, y_train)

                if i == 0:
                    tree = tree_clf.trained_tree
                    visualiser = VisualiseTree(tree, pruning)

                #predict on test data
                self.y_predicted = tree_clf.predict(x_test)
                self.y = y_test

                #evaluation metrics
                confusion_mat = self.get_confusion_matrix() 
                accuracy = self.get_accuracy()
                precision, recall, f1_score = zip(*self.get_precision_recall_f1())

                metrics = (confusion_mat, accuracy, precision, recall, f1_score)

                for i, metric in enumerate(list(metrics_dict.keys())):
                    if metric == "Confusion Matrices":
                        metrics_dict[metric].append(np.array(metrics[i]))
                    else:
                        metrics_dict[metric].append(metrics[i])

            #calculate average metrics for cross-validation
            avg_metrics = self.average_metrics(metrics_dict, pruning)

            #print final metrics
            self.print_metrics(avg_metrics, pruning)

        #if we are pruning, perform nested cross-validation to evaluate pruned trees
        else: 
            outer_metrics = {"Confusion Matrices": [], "Accuracies": [], "Precisions": [], 
                             "Recalls": [], "F1 Scores": [],  "Max Depth Before": [], "Max Depth After": []}

            # Outer CV (10-fold)
            outer_folds = 10 
            cv.folds = outer_folds
            print("Currently evaluating: ")
            for i, (trainval_indices, test_indices) in enumerate(cv.train_test_k_fold(x, y)):
                print("\nOUTER FOLD #", i, "\n")
                x_trainval = x[trainval_indices, :]
                y_trainval = y[trainval_indices]
                x_test = x[test_indices, :]
                y_test = y[test_indices]

                # Pre-split data for inner cross-validation (9 inner folds)
                inner_folds = 9
                cv.folds = inner_folds
                splits = cv.train_test_k_fold(x_trainval, y_trainval)

                inner_metrics = outer_metrics.copy() 

                # Inner CV (10-fold again)  
                for j, (train_indices, val_indices) in enumerate(splits):
                    print("Inner Fold #", j)
                    #retrieve training and validation sets from random indices (splits)
                    x_train = x_trainval[train_indices, :]
                    y_train = y_trainval[train_indices]
                    x_val = x_trainval[val_indices, :]
                    y_val = y_trainval[val_indices]

                    #fit decision tree classifier on training data
                    tree_clf = DecisionTreeClassifier(max_depth=100)
                    tree_clf.fit(x_train, y_train)

                     #prune tree and get maximum depth before and after pruning
                    tree = tree_clf.trained_tree
                    tree_prune = TreePruning(tree, x_train, x_val, y_train, y_val, self)
                    max_depth_before, max_depth_after, post_pruned = tree_prune.prune_tree()

                    if i == 0 and j == 0:
                        visualiser = VisualiseTree(post_pruned, pruning)

                    self.y_predicted = tree_prune.predict_tree(x_test, post_pruned) #evaluate pruned tree on test set
                    self.y = y_test
                    
                    #evaluation metrics for each inner fold 
                    confusion_mat = self.get_confusion_matrix() 
                    accuracy = self.get_accuracy()
                    precision, recall, f1_score = zip(*self.get_precision_recall_f1())

                    #make list of all metrics
                    metrics = (confusion_mat, accuracy, precision, recall, f1_score, max_depth_before, max_depth_after)

                    for i, metric in enumerate(list(inner_metrics.keys())):
                        if metric == "Confusion Matrices":
                            inner_metrics[metric].append(np.array(metrics[i]))
                        else: 
                            inner_metrics[metric].append(metrics[i])

                #calculate final average metrics for nested cross-validation from averages of inner folds
                avg_inner_metrics =  self.average_metrics(inner_metrics, pruning)

                for i, metric in enumerate(list(outer_metrics.keys())):
                    if metric == "Confusion Matrices":
                        outer_metrics[metric].append(np.array(avg_inner_metrics[i]))
                    else: 
                        outer_metrics[metric].append(avg_inner_metrics[i])

            # calculate average metrics for outer folds
            avg_outer_metrics =  self.average_metrics(outer_metrics, pruning)

            # print final metrics
            self.print_metrics(avg_outer_metrics, pruning)
        
        #plot sample classifier
        visualiser.plot()
