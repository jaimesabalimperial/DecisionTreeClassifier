import numpy as np
from evaluation.cross_validation import CrossValidation
from classifier.tree import DecisionTreeClassifier
from evaluation.prune import TreePruning


class EvaluationMetrics:
    def __init__(self):
        self.y = None
        self.y_predicted = None
        self.k = None

    def compute_confusion_matrix(self):
        """
        """
        # compute TP
        num_of_classes = len(list(set(self.y)))
        confusion_mat = np.empty([num_of_classes, num_of_classes])
        for row, true_class in enumerate(list(set(self.y))):
            for col, pred_class in enumerate(list(set(self.y))):
                confusion_mat[row][col] = sum((self.y == true_class) & (self.y_predicted == pred_class))
        return confusion_mat

    def compute_accuracy(self):
        """
        """
        assert len(self.y) == len(self.y_predicted)
        try:
            return np.sum(self.y == self.y_predicted) / len(self.y)
        except ZeroDivisionError:
            return 0

    def compute_precision_recall_f1(self):
        """
        """
        confusion_matrix = self.compute_confusion_matrix()
        precision = []
        recall = []
        f1_score = []
        for room in range(len(list(set(self.y)))):
            precision.append(confusion_matrix[room][room] / np.array(confusion_matrix).sum(axis=1)[room])
            recall.append(confusion_matrix[room][room] / np.array(confusion_matrix).sum(axis=0)[room])
            curr_f1 = (2 * precision[room] * recall[room]) / (precision[room] + recall[room])
            f1_score.append(curr_f1)
        return precision, recall, f1_score

    def compute_averaged_f1(self):
        """
        """
        confusion_matrix = self.compute_confusion_matrix()
        precision = []
        recall = []
        f1_score = []
        for room in range(len(list(set(self.y)))):
            try:
                precision.append(confusion_matrix[room][room] / np.array(confusion_matrix).sum(axis=1)[room])
                recall.append(confusion_matrix[room][room] / np.array(confusion_matrix).sum(axis=0)[room])
                curr_f1 = (2 * precision[room] * recall[room]) / (precision[room] + recall[room])
            except ZeroDivisionError:
                curr_f1 = 0
            f1_score.append(curr_f1)
        return sum(f1_score)/len(f1_score)

    def average_metrics(self, metrics, pruning):
        """
        """
        #compute average of all folds
        total_confusion_mat = sum(metrics["Confusion Matrices"])
        average_accuracy = np.mean(metrics["Accuracies"])
        average_precision = list(map(np.mean, zip(*metrics["Precisions"])))
        average_recall = list(map(np.mean, zip(*metrics["Recalls"])))
        average_f1_score = list(map(np.mean, zip(*metrics["F1 Scores"])))


        if pruning: 
            average_max_depth_before = np.mean(metrics["Max Depth Before"])
            average_max_depth_after = np.mean(metrics["Max Depth After"])

            return (total_confusion_mat, average_accuracy, average_precision, 
                    average_recall , average_f1_score, average_max_depth_before, average_max_depth_after)

    def compute_CV_results(self, cv, x, y, pruning):
        """
        """
        if not pruning:
            #initialise metric lists
            metrics = {"Confusion Matrices": [], "Accuracies": [], "Precisions": [], "Recalls": [], "F1 Scores": []}

            for i, (train_indices, test_indices) in enumerate(cv.train_test_k_fold(x, y)):
                # get the dataset from the correct splits
                x_train = x[train_indices, :]
                y_train = y[train_indices]
                x_test = x[test_indices, :]
                y_test = y[test_indices]

                tree_clf = DecisionTreeClassifier(max_depth=100)
                tree_clf.fit(x_train, y_train)

                #predict on test data
                self.y_predicted = tree_clf.predict(x_test)
                self.y = y_test

                #evaluation metrics
                confusion_mat = self.compute_confusion_matrix() 
                accuracy = self.compute_accuracy()
                precision, recall, f1_score = self.compute_precision_recall_f1()

                metrics["Confusion Matrices"].append(np.array(confusion_mat))
                metrics["Accuracies"].append(accuracy)
                metrics["Precisions"].append(precision)
                metrics["Recalls"].append(recall)
                metrics["F1 Scores"].append(f1_score)

            #calculate average metrics for cross-validation
            self.average_metrics(metrics, pruning)

        else: 
            outer_metrics = {"Confusion Matrices": [], "Accuracies": [], "Precisions": [], 
                             "Recalls": [], "F1 Scores": [],  "Max Depth Before": [], "Max Depth After": []}

            # Outer CV (10-fold)
            for i, (trainval_indices, test_indices) in enumerate(cv.train_test_k_fold(x, y)):
                print("\nOuter Fold ", i)
                x_trainval = x[trainval_indices, :]
                y_trainval = y[trainval_indices]
                x_test = x[test_indices, :]
                y_test = y[test_indices]
                self.y = y_test

                # Pre-split data for inner cross-validation 
                splits = cv.train_test_k_fold(x_trainval, y_trainval)

                
                inner_metrics = outer_metrics.copy()

                # Inner CV (10-fold again)  
                for j, (train_indices, val_indices) in enumerate(splits):
                    print("Inner Fold ", j)
                    x_train = x_trainval[train_indices, :]
                    y_train = y_trainval[train_indices]
                    x_val = x_trainval[val_indices, :]
                    y_val = y_trainval[val_indices]

                    tree_clf = DecisionTreeClassifier(max_depth=100)
                    tree_clf.fit(x_train, y_train)

                    tree = tree_clf.trained_tree
                    tree_prune = TreePruning(tree, x_train, x_val, y_train, y_val, self)
                    max_depth_before, max_depth_after, post_pruned = tree_prune.prune_tree()

                    self.y_predicted = tree_prune.predict_tree(x_val, post_pruned)
                    
                    #evaluation metrics
                    confusion_mat = self.compute_confusion_matrix() 
                    accuracy = self.compute_accuracy()
                    precision, recall, f1_score = self.compute_precision_recall_f1()

                    inner_metrics["Confusion Matrices"].append(np.array(confusion_mat))
                    inner_metrics["Accuracies"].append(accuracy)
                    inner_metrics["Precisions"].append(precision)
                    inner_metrics["Recalls"].append(recall)
                    inner_metrics["F1 Scores"].append(f1_score)
                    inner_metrics["Max Depth Before"].append(max_depth_before)
                    inner_metrics["Max Depth After"].append(max_depth_after)

                #calculate average metrics for cross-validation
                (total_confusion_mat, average_accuracy, average_precision, 
                average_recall , average_f1_score, avg_max_depth_before, avg_max_depth_after) =  self.average_metrics(inner_metrics, pruning)

                outer_metrics["Confusion Matrices"].append(total_confusion_mat)
                outer_metrics["Accuracies"].append(average_accuracy)
                outer_metrics["Precisions"].append(average_precision)
                outer_metrics["Recalls"].append(average_recall)
                outer_metrics["F1 Scores"].append(average_f1_score)
                outer_metrics["Max Depth Before"].append(avg_max_depth_before)
                outer_metrics["Max Depth After"].append(avg_max_depth_after)


            # calculate average metrics for outer folds
            (final_total_confusion_mat, final_average_accuracy, 
            final_average_precision, final_average_recall , 
            final_average_f1_score, final_average_max_depth_before,
            final_average_max_depth_after) =  self.average_metrics(outer_metrics, pruning)

            # normalize confusion matrix
            sum_of_rows = final_total_confusion_mat.sum(axis=1)
            normalized_confusion_matrix = final_total_confusion_mat / sum_of_rows[:, np.newaxis]

            # print final metrics
            print(f"\nThe total confusion matrix is: \n {normalized_confusion_matrix}")
            print(f"\nThe average accuracy is: \n{final_average_accuracy}")
            print(f"\nThe average precision for each class is: \n{final_average_precision}")
            print(f"\nThe average recall for each class is: \n{final_average_recall}")
            print(f"\nThe average f1 score for each class is: \n{final_average_f1_score}")
            print(f"\nThe average maximum depth before pruning is : \n {final_average_max_depth_before}")
            print(f"\nThe average maximum depth after pruning is : \n {final_average_max_depth_after}")

            
            

            

    def evaluate_CV(self, x, y, pruning = False):
        """
        """
        cv = CrossValidation()
        self.compute_CV_results(cv, x, y, pruning)