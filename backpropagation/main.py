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


python main.py ../data/configs/network2.txt ../data/configs/initial_weights2.txt ../data/datasets/wine.txt
"""

def main():
    config_file = sys.argv[1]
    initial_weights_file = sys.argv[2]
    dataset_file = sys.argv[3]

    fileUtils = FileUtils(dataset_file=dataset_file, config_file=config_file)
    dataset = fileUtils.getDataset()

    # neurons_per_layer = [1, 2, 1]
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
    # network.verify()


if __name__ == '__main__':
    main()
