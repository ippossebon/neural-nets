import csv
import random
import math
import statistics
import jsonpickle
import pprint
import sys
import numpy as np

# import matplotlib.pyplot as plt
from utils import FileUtils
from neuralnetwork import NeuralNetwork

"""
./backpropagation network.txt initial_weights.txt dataset.txt
"""

print_multi_array = pprint.PrettyPrinter(indent=4)

def main():
    config_file = './data/configs/network2.txt'
    initial_weights_file = './data/configs/initial_weights2.txt'
    dataset_file = './data/datasets/wine.txt'

    fileUtils = FileUtils(dataset_file=dataset_file, config_file=config_file)
    dataset = fileUtils.getDataset()

    #normalized_dataset = normalizeDataset(dataset)

    neurons_per_layer = [2, 4, 3, 2]
    network = NeuralNetwork(
        config_file=config_file,
        dataset=dataset,
        initial_weights_file=initial_weights_file,
        neurons_per_layer=neurons_per_layer
    )

    # network.backpropagation()
    weights = network.runNetwork(max_iter=250)

    print('')
    print('Pesos corretos = ')
    print(weights)

    # print('---- Verifica -----')
    #
    # network.verify()


def runCrossValidation():
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


# Normaliza as features dado o limite [0,1]
def normalizeDataset(dataset):
    for i in range(len(dataset)):
        dataset_line = dataset[i].attributes

        min_value = np.min(dataset_line)
        max_value = np.max(dataset_line)

        # calcula novo valor normalizado para cada parametro
        for j in range(len(dataset_line)):
            dataset_line[j] = (dataset_line[j] - min_value)/(max_value - min_value)


    # for i in range(len(dataset)):
    #     dataset_line = dataset[i].attributes
    #
    #     # calcula novo valor normalizado para cada parametro
    #     for j in range(len(dataset_line)):
    #         #print(len(dataset_line))
    #         ex1 = Instance(attributes=dataset_line[j], classification=dataset[i].classification)
    #         #print(self.dataset[i].classification)
    #         training_data.append(ex1)
    #
    # #print(training_data)

# Cria o conjunto de bootstrap
def getBootstrap(data_set, size):
    bootstrap = []

    for i in range(size):
        index = random.randint(0, len(data_set)-1)
        bootstrap.append(data_set[index])

    return bootstrap


# Executa a validacao cruzada estratificada
def crossValidation(attributes, attributes_types, target_class, folds, b, k):

    config_file = './data/configs/network.txt'
    initial_weights_file = './data/configs/initial_weights.txt'
    dataset_file = './data/datasets/wine.txt'

    accuracy_values = []
    precision_values = []
    recall_values = []
    fmeasure_values = []

    for i in range(k):
        training_set_folds = list(folds)
        training_set_folds.remove(folds[i])
        training_set = transformToList(training_set_folds)

        # bootstrap tem o tamanho do conjunto de treinamento
        bootstrap_size = len(training_set)

        test_set = folds[i]
        forest = []

        for j in range(b):
            bootstrap = getBootstrap(training_set, bootstrap_size)
            neurons_per_layer = [1, 2, 1]

            network = NeuralNetwork(
                config_file=config_file,
                dataset_file=dataset_file,
                initial_weights_file=initial_weights_file,
                neurons_per_layer=neurons_per_layer
            )

            network.backpropagation()

            netgroup.append(network)

        # Usa o ensemble de B redes neurais para prever as instancias do fold i
        # (fold de teste) e avaliar desempenho do algoritmo
        true_positives, false_positives, false_negatives, true_negatives = evaluateNetgroup(netgroup, test_set, target_class)

        accuracy_values.append(calculateAccuracy(true_positives, true_negatives, false_positives, false_negatives))

        precision_value = calculatePrecision(true_positives, false_positives)
        precision_values.append(precision_value)

        recall_value = calculateRecall(true_positives, false_negatives)
        recall_values.append(recall_value)
        fmeasure_values.append(calculateF1Measure(precision_value, recall_value))

    accuracy = sum(accuracy_values)/len(accuracy_values)
    precision = sum(precision_values)/len(precision_values)
    recall = sum(recall_values)/len(recall_values)
    fmeasure = sum(fmeasure_values)/len(fmeasure_values)

    return accuracy, precision, recall, fmeasure

def calculateAccuracy(true_positives, true_negatives, false_positives, false_negatives):
    return float((true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives))

def calculateRecall(true_positives, false_negatives):
    return float((true_positives)/(true_positives + false_negatives))

def calculatePrecision(true_positives, false_positives):
    return float((true_positives)/(true_positives + false_positives))

def calculateF1Measure(precision, recall):
    return float((2*precision*recall)/(precision+recall))

def evaluateNetgroup(forest, test_set, target_class):
    instances_copy = list(test_set)

    class_distinct_values = getClassDistinctValues(target_class, test_set)

    true_positives = 0
    false_positives = {}
    true_negatives = {}
    false_negatives = {}

    for value in class_distinct_values:
        false_positives[value] = 0
        true_negatives[value] = 0
        false_negatives[value] = 0

    predictions = []
    correct_classes = []

    # Para cada instância do conjunto de validação
    for instance in instances_copy:
        correct_class = instance[target_class]
        correct_classes.append(correct_class)

        predicted_class = netgroupPredict(netgroup, instance)

        predictions.append(predicted_class)

    for i in range(len(predictions)):
        if predictions[i] == correct_classes[i]:
            true_positives = true_positives + 1
        else:
            for value in class_distinct_values:
                if correct_classes[i] == value and predictions[i] != value:
                    # falso negativo
                    false_negatives[value] = false_negatives[value] + 1
                elif correct_classes[i] != value and predictions[i] == value:
                    # falso positivo
                    false_positives[value] = false_positives[value]  + 1
                elif correct_classes[i] != value and predictions[i] != value:
                    # verdadeiro negativo
                    true_negatives[value] = true_negatives[value] + 1


    avg_false_positives = getAverageValue(false_positives)
    avg_false_negatives = getAverageValue(false_negatives)
    avg_true_negatives = getAverageValue(true_negatives)

    return true_positives, avg_false_positives, avg_false_negatives, avg_true_negatives


def getAverageValue(values_dict):
    values_list = []
    classes_count = 0

    for value in values_dict:
        values_list.append(values_dict[value])
        classes_count = classes_count + 1

    avg_value = float(sum(values_list) / classes_count)
    return avg_value

def netgroupPredict(netgroup, instance):
    predictions = []

    for network in netgroup:
        predictions.append(network.classify(instance))

    most_frequent_class = max(set(predictions), key=predictions.count)

    return most_frequent_class

if __name__ == '__main__':
    main()
