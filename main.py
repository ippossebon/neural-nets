import csv
import random
import math
import statistics
import jsonpickle
import pprint
import matplotlib.pyplot as plt

from neuralnetwork import NeuralNetwork

"""
./backpropagation network.txt initial_weights.txt dataset.txt
"""

print_multi_array = pprint.PrettyPrinter(indent=4)

def main():
    # config_file = './data/configs/network.txt'
    # initial_weights_file = './data/configs/initial_weights.txt'
    # dataset_file = './data/datasets/wine.txt'
    #
    # neurons_per_layer = [1, 2, 1]
    # network = NeuralNetwork(
    #     config_file=config_file,
    #     dataset_file=dataset_file,
    #     initial_weights_file=initial_weights_file,
    #     neurons_per_layer=neurons_per_layer
    # )
    #
    # network.backpropagation()

    results_accuracy = []
    results_precision = []
    results_recall = []
    results_f1measure = []

    k = 10
    x = range(1,50)
    x_axis = [1, 5, 10, 20, 25, 30, 35, 40, 45, 50]

    for i in x_axis:
        folds = getKStratifiedFolds(instances, target_class, k=k)
        results = crossValidation(attributes,
                                attributes_types,
                                target_class,
                                folds,
                                b=1,
                                k=k)

        results_accuracy.append(results[0])
        results_precision.append(results[1])
        results_recall.append(results[2])
        results_f1measure.append(results[3])

    plt.xticks(np.arange(min(x), max(x)+1, 5.0))
    plt.plot(x_axis, results_accuracy, label = "Accuracy")
    plt.plot(x_axis, results_precision, label = "Precision")
    plt.plot(x_axis, results_recall, label = "Recall")
    plt.plot(x_axis, results_f1measure, label = "F1-Measure")
    plt.ylabel('Values')
    plt.xlabel('Number of trees')
    plt.title('Results for' + file_name)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
