import numpy as np
from evaluation.cross_validation import CrossValidation
from classifier.tree import DecisionTreeClassifier


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

    def compute_CV_results(self, X_train, y_train, X_test, y_test):
                #initialise metric lists
        confusion_matrices = []
        accuracies_list = []
        precisions_list = []
        recalls_list = []
        f1_scores = []

        for i in range(self.k):
            # build tree
            tree_clf = DecisionTreeClassifier(max_depth=100)
            tree_clf.fit(X_train[i], y_train[i])
            self.y_predicted = tree_clf.predict(X_test[i])
            self.y = y_test[i]
            confusion_mat = self.compute_confusion_matrix() 
            accuracy = self.compute_accuracy()
            precision, recall, f1_score = self.compute_precision_recall_f1()

            confusion_matrices.append(np.array(confusion_mat))
            accuracies_list.append(accuracy)
            precisions_list.append(precision)
            recalls_list.append(recall)
            f1_scores.append(f1_score)

        #compute average of all folds
        total_confusion_mat = sum(confusion_matrices)
        average_accuracy = np.mean(accuracies_list)
        average_precision = list(map(np.mean, zip(*precisions_list)))
        average_recall = list(map(np.mean, zip(*recalls_list)))
        average_f1_score = list(map(np.mean, zip(*f1_scores)))

        print(f"\nThe total confusion matrix is: \n {total_confusion_mat}")
        print(f"\nThe average accuracy is: \n{average_accuracy}")
        print(f"\nThe average precision for each class is: \n{average_precision}")
        print(f"\nThe average recall for each class is: \n{average_recall}")
        print(f"\nThe average f1 score for each class is: \n{average_f1_score}")

        


    def evaluate_CV(self, X, y, nested_CV = False):
        """
        """
        cv = CrossValidation(X, y)
        self.k = cv.folds

        if nested_CV == False:
            X_train, y_train, X_test, y_test = cv.generate_CV_data()
            self.compute_CV_results(X_train, y_train, X_test, y_test)
        else: 
            pass
        


