import sys
from evaluation.evaluation_metrics import EvaluationMetrics
from data_manipulation.load_dataset import load_dataset

def print_results(data, pruning):
    
    initial_message = {("Clean", True): "\nPerforming nested cross-validation on pruned tree classifier trained on clean data:\n",
                       ("Clean", False): "\nPerforming cross-validation on tree classifier trained on clean data:\n",
                       ("Noisy", True): "\nPerforming nested cross-validation on pruned tree classifier trained on noisy data:\n",
                       ("Noisy", False): "\nPerforming cross-validation on tree classifier trained on noisy data:\n",
                      }
    #evaluation class
    metrics = EvaluationMetrics()

    if data == "Clean": 
        #make initial prediction on test set for clean data
        filepath = 'wifi_db/clean_dataset.txt'
        x, y = load_dataset(filepath, clean = True)


    elif data == "Noisy":
        filepath = 'wifi_db/noisy_dataset.txt'
        x, y = load_dataset(filepath, clean = False)

    print(initial_message[(data,pruning)])

    if pruning:
        #perform nested cross-validation for pruned trees
        metrics.evaluate_CV(x, y, pruning=True)
    else:
        #perform cross-validation evaluation without pruning
        metrics.evaluate_CV(x, y, pruning=False)

if __name__ == '__main__':

    clean_argv_list = ["clean", "Clean", "CLEAN", "c", "C"]
    noisy_argv_list = ["noisy", "Noisy", "NOISY", "n", "N"]
    pruning_argv_list = ["prune", "pruning", "Prune", "p", "P"]


    if sys.argv[1] in noisy_argv_list:
        if len(list(sys.argv)) == 2:
            print_results(data = "Noisy", pruning = False)
        elif sys.argv[2] in pruning_argv_list:
            print_results(data = "Noisy", pruning = True)

    elif sys.argv[1] in clean_argv_list:
        if len(list(sys.argv)) == 2:
            print_results(data = "Clean", pruning = False)

        elif sys.argv[2] in pruning_argv_list:
            print_results(data = "Clean", pruning = True)
    else:
        print("\nCommand not understood, examples of acceptable commands are: \n")
        print("python3 main.py clean -----> perform cross-validation on tree classifier for clean dataset.\n")
        print("python3 main.py clean prune -----> perform nested cross-validation on pruned tree classifier for clean dataset.\n")
        print("python3 main.py noisy -----> perform cross-validation on tree classifier for noisy dataset.\n")
        print("python3 main.py noisy prune -----> perform nested cross-validation on pruned tree classifier for noisy dataset.\n")

