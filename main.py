import csv
import random
import math
import statistics
import jsonpickle
import pprint

# import matplotlib.pyplot as plt
from utils import FileUtils
from neuralnetwork import NeuralNetwork

"""
./backpropagation network.txt initial_weights.txt dataset.txt
"""

print_multi_array = pprint.PrettyPrinter(indent=4)

def main():
    config_file = './data/configs/network.txt'
    initial_weights_file = './data/configs/initial_weights.txt'
    dataset_file = './data/datasets/wine.txt'

    fileUtils = FileUtils(dataset_file=dataset_file, config_file=config_file)
    dataset = fileUtils.getDataset()

    #normalized_dataset = normalizeDataset(dataset)

    neurons_per_layer = [1, 2, 1]
    # neurons_per_layer = [2, 4, 3, 2]
    network = NeuralNetwork(
        config_file=config_file,
        dataset=dataset,
        initial_weights_file=initial_weights_file,
        neurons_per_layer=neurons_per_layer
    )

    network.backpropagation()



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


    for i in range(len(dataset)):
        dataset_line = dataset[i].attributes

        # calcula novo valor normalizado para cada parametro
        for j in range(len(dataset_line)):
            #print(len(dataset_line))
            ex1 = Instance(attributes=dataset_line[j], classification=dataset[i].classification)
            #print(self.dataset[i].classification)
            training_data.append(ex1)

    #print(training_data)



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

            forest.append(network)

        # Usa o ensemble de B arvores para prever as instancias do fold i
        # (fold de teste) e avaliar desempenho do algoritmo
        true_positives, false_positives, false_negatives, true_negatives = evaluateForest(forest, test_set, target_class)

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


if __name__ == '__main__':
    main()
