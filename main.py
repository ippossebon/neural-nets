import csv
import random
import math
import statistics
import jsonpickle
import pprint

from neuralnetwork import NeuralNetwork

"""
./backpropagation network.txt initial_weights.txt dataset.txt
"""

print_multi_array = pprint.PrettyPrinter(indent=4)

def main():
    config_file = './data/configs/network.txt'
    initial_weights_file = './data/configs/initial_weights.txt'
    dataset_file = './data/datasets/wine.txt'

    neurons_per_layer = [1, 2, 1]
    network = NeuralNetwork(
        config_file=config_file,
        dataset_file=dataset_file,
        initial_weights_file=initial_weights_file,
        neurons_per_layer=neurons_per_layer
    )

    network.backpropagation()

if __name__ == '__main__':
    main()
