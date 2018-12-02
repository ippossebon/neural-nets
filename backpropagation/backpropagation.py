import csv
import random
import math
import sys
import numpy as np

from utils import FileUtils
from neuralnetwork import NeuralNetwork

"""
./backpropagation network.txt initial_weights.txt dataset.txt

python backpropagation.py ../data/configs/network.txt ../data/configs/initial_weights.txt ../data/datasets/wine.txt
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

    # Verifica a quantidade de classes distintas
    classes = []
    for instance in dataset:
        if instance.classification not in classes:
            classes.append(instance.classification)

    # Inicializa rede com o número de atributos de cada instância como número de
    # neurônios na primeira camada e número de classes possíveis
    # como quantidade de neurônios da última camada
    neurons_first_layer = len(dataset[0].attributes)
    neurons_last_layers = len(classes)

    neurons_per_layer = [neurons_first_layer, neurons_last_layers]

    index = 1

    for layer in hidden_layers:
        neurons_per_layer.insert(index, layer)
        index+=1

    regularization_factor = config[0]
    print('Parametro de regularizacao lambda={0}\n'.format(regularization_factor))

    neurons_per_layer_without_bias = list(neurons_per_layer)
    for i in range(len(neurons_per_layer)):
        if i == 0:
            neurons_per_layer_without_bias[i] = neurons_per_layer[i]
        else:
            neurons_per_layer_without_bias[i] = neurons_per_layer[i] - 1

    print('Inicializando rede com a seguinte estrutura de neuronios por camada: {0}\n'.format(neurons_per_layer_without_bias))

    network = NeuralNetwork(
        config_file = config_file,
        dataset = dataset,
        initial_weights_file = initial_weights_file,
        neurons_per_layer = neurons_per_layer_without_bias,
        reg_factor = regularization_factor
    )

    print('\n\n--------------------------------------------')
    print('Rodando backpropagation')

    network.runBackpropagation()
    network.checkGradiends()


if __name__ == '__main__':
    main()
