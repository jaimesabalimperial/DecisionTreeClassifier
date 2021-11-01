from evaluation.evaluation_metrics import EvaluationMetrics
from data_manipulation.load_dataset import load_dataset

def print_results(data = "Clean"):
    
    #evaluation class
    metrics = EvaluationMetrics()

    if data == "Clean": 
        #make initial prediction on test set for clean data
        filepath = 'wifi_db/clean_dataset.txt'
        x, y = load_dataset(filepath, clean = True)

        #perform cross-validation evaluation pre-pruning
        metrics.evaluate_CV(x, y, pruning=True)


    elif data == "Noisy":
        filepath = 'wifi_db/noisy_dataset.txt'
        x, y = load_dataset(filepath, clean = False)

        #perform cross-validation evaluation pre-pruning
        metrics.evaluate_CV(x, y, pruning=True)




    #tree = tree_clf
    #tree_prune = TreePruning(tree, X_train, X_test, y_train, y_train)
    #tree_prune.main_prune()


if __name__ == '__main__':
    print_results(data = "Noisy")

