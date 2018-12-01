import csv
import random
import math
import sys
import numpy as np

from utils import FileUtils
from neuralnetwork import NeuralNetwork

"""
./backpropagation network.txt initial_weights.txt dataset.txt

config_file = '../data/configs/network2.txt'
initial_weights_file = '../data/configs/initial_weights2.txt'
dataset_file = '../data/datasets/wine.txt'


python main.py ../data/configs/network.txt ../data/configs/initial_weights.txt ../data/datasets/wine.txt
"""

def main():
    config_file = sys.argv[1]
    initial_weights_file = sys.argv[2]
    dataset_file = sys.argv[3]

    fileUtils = FileUtils(dataset_file=dataset_file)
    dataset = fileUtils.getDataset()

    hidden_layers = []

    fileUtils = FileUtils(config_file=config_file)
    config = fileUtils.getConfigParams()

    config_len = len(config)

    for layer in range(2,config_len-1):
        hidden_layers.append(config[layer])

    reg_factor = config[0]

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

    neurons_per_layer = [neurons_first_layer, neurons_last_layers]

    index = 1

    for layer in hidden_layers:
        neurons_per_layer.insert(index, layer)
        index+=1

    network = NeuralNetwork(
        config_file = config_file,
        dataset = dataset,
        initial_weights_file = initial_weights_file,
        neurons_per_layer = neurons_per_layer,
        reg_factor = config[0]
    )

    network.backpropagation()
    weights = network.runNetwork(max_iter=50)

    print('')
    print('Pesos corretos = ')
    print(weights)

    # print('\n \n \n')
    # network.checkGradient()

# Normaliza as features dado o limite [0,1]
def normalizeDataset(dataset):
    for i in range(len(dataset)):
        dataset_line = dataset[i].attributes

        min_value = np.min(dataset_line)
        max_value = np.max(dataset_line)

        # calcula novo valor normalizado para cada parametro
        for j in range(len(dataset_line)):
            dataset_line[j] = (dataset_line[j] - min_value)/(max_value - min_value)

if __name__ == '__main__':
    main()
