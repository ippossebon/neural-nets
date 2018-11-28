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

# config_file = '../data/configs/network2.txt'
# initial_weights_file = '../data/configs/initial_weights2.txt'
# dataset_file = '../data/datasets/ionosphere.txt'
# fileUtils = FileUtils(dataset_file=dataset_file)
# dataset = fileUtils.getDataset()
# target_class = 'class'

network_results = {}

def main():
    dataset_file = '../data/datasets/wine.txt'
    fileUtils = FileUtils(dataset_file=dataset_file)
    dataset = fileUtils.getDataset()
    target_class = 'class'
    normalizeDataset(dataset)

    # Verifica a quantidade de classes distintas
    classes = []
    for instance in dataset:
        if instance.classification not in classes:
            classes.append(instance.classification)

    # Altera o dataset para ter o numero de saidas como o numero de classes possiveis
    for i in range(len(dataset)):
        classification = dataset[i].classification
        classification_dict = {}

        for c in classes:
            if c == classification:
                # classificação da instancia
                classification_dict[c] = 1
            else:
                classification_dict[c] = 0

        dataset[i].classification = classification_dict

    # Inicializa rede com o número de atributos de cada instância como número de
    # neurônios na primeira camada e número de classes possíveis
    # como quantidade de neurônios da última camada
    neurons_first_layer = len(dataset[i].attributes)
    neurons_last_layers = len(classes)
    neurons_per_layer = [neurons_first_layer, 4, 3, neurons_last_layers]

    network = NeuralNetwork(
        dataset=dataset,
        reg_factor = 0.25,
        neurons_per_layer=neurons_per_layer
    )

    # network.backpropagation()
    weights = network.runNetwork(max_iter=50)

    print('')
    print('Pesos corretos = ')
    print(weights)

    # print('---- Verifica -----')
    #
    # network.verify()

# ## Executando crossValidation para obter melhor arquitetura de rede para um dataset ##
    results_accuracy = []
    results_precision = []
    results_recall = []
    results_f1measure = []

    neurons_list = []
    reg_factor_list = []
    k = 10
    n = 5
    r = 0.21
    x = range(1,50)
    x_axis = [1, 5, 10, 20, 25, 30, 35, 40, 45, 50]

    for i in x_axis:
        folds = getKStratifiedFolds(dataset, target_class, k=k)
        results = crossValidation(target_class,
                                folds,
                                neurons=n,
                                reg_factor=r,
                                b=1,
                                k=k)

        results_accuracy.append(results[0])
        results_precision.append(results[1])
        results_recall.append(results[2])
        results_f1measure.append(results[3])

    # plt.xticks(np.arange(min(x), max(x)+1, 5.0))
    # plt.plot(x_axis, results_accuracy, label = "Accuracy")
    # plt.plot(x_axis, results_precision, label = "Precision")
    # plt.plot(x_axis, results_recall, label = "Recall")
    # plt.plot(x_axis, results_f1measure, label = "F1-Measure")
    # plt.ylabel('Values')
    # plt.xlabel('Number of neurons/regularization factor')
    # plt.title('Results for' + file_name)
    # plt.legend()
    # plt.show()

    #####################################################################################

    ## Obtem os indices das melhores medidas ##

    best_accuracy = np.max(results_accuracy)
    best_precision = np.max(results_precision)
    best_recall = np.max(results_recall)
    best_f1measure = np.max(results_f1measure)

    accuracy_index = results_accuracy.index(best_accuracy)
    precision_index = results_precision.index(best_precision)
    recall_index = results_recall.index(best_recall)
    f1measure_index = results_f1measure.index(best_f1measure)

    best_neuron_index = accuracy_index
    best_reg_factor_index = accuracy_index

    # Chamar o crossValidation novamente com os melhores valores
    # mantendo a arquitetura da rede fixa mas variando o numero
    # de exemplos

# Normaliza as features dado o limite [0,1]
def normalizeDataset(dataset):
    for i in range(len(dataset)):
        dataset_line = dataset[i].attributes

        min_value = np.min(dataset_line)
        max_value = np.max(dataset_line)

        # calcula novo valor normalizado para cada parametro
        for j in range(len(dataset_line)):
            dataset_line[j] = (dataset_line[j] - min_value)/(max_value - min_value)

# Cria o conjunto de bootstrap
def getBootstrap(data_set, size):
    bootstrap = []

    for i in range(size):
        index = random.randint(0, len(data_set)-1)
        bootstrap.append(data_set[index])

    return bootstrap

# Executa a validacao cruzada estratificada
def crossValidation(target_class, folds, neurons, reg_factor, b, k):

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
        netgroup = []

        for j in range(b):
            bootstrap = getBootstrap(training_set, bootstrap_size)
            neurons_per_layer = [1, neurons, 1]

            network = NeuralNetwork(
                dataset_file=bootstrap,
                neurons_per_layer=neurons_per_layer,
                reg_factor=reg_factor
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

    results_list = []

    results_list.append(accuracy)
    results_list.append(precision)
    results_list.append(recall)
    results_list.append(fmeasure)

    # Adiciona resultados no dicionario respectivo a rede
    network_results[network] = results_list

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

def getKStratifiedFolds(data_set, target_class, k):
    instances_by_class = getClassesSubsets(target_class, data_set)
    folds = [None] * k

    # Inicializa a lista de folds
    for i in range(k):
        folds[i] = []

    for class_value in instances_by_class:
        # pega K valores deste subset
        instance_index = 0
        for instance in instances_by_class[class_value]:
            fold_index = instance_index % k
            folds[fold_index].append(instance)
            instance_index = instance_index + 1

    return folds

def getClassesSubsets(target_class, data):
    distinct_values = getClassDistinctValues(target_class, data)
    class_subsets = {}
    for value in distinct_values:
        for instance in data:
            if instance.classification == value:
                for key in value:
                    if key not in class_subsets:
                        class_subsets[key] = []
                    class_subsets[key].append(instance)

    return class_subsets

def getClassDistinctValues(target_class, data):
    distinct_values = []
    for instance in data:
        if instance.classification not in distinct_values:
            distinct_values.append(instance.classification)

    return distinct_values

def transformToList(list_of_lists):
    new_list = []
    for l in list_of_lists:
        for value in l:
            new_list.append(value)

    return new_list

if __name__ == '__main__':
    main()
