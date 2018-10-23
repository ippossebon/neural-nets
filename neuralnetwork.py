import math
import numpy as np

from instance import Instance
from utils import FileUtils


class NeuralNetwork(object):

    def __init__(self, initial_weights_file, neurons_per_layer, dataset_file):
        self.dataset_file = dataset_file
        self.initial_weights_file = initial_weights_file
        self.output     = []
        self.prediction = []
        self.dataset = None
        self.num_layers = len(neurons_per_layer)
        self.neurons_per_layer = neurons_per_layer
        self.num_input = None
        self.num_hidden = None
        self.num_output = None

        self.training_data = None

        self.theta = []
        self.initTheta()

        self.activation = []
        self.initActivationsVector()

        self.setDataset()
        self.setTrainingData()

    def initTheta(self):
        for i in range(len(self.neurons_per_layer) - 1):
            # theta é uma lista de matrizes - contém L-1 items
            # linhas = número de neuronios da proxima camada, colunas = número
            # de neuronios da camada atual + bias
            self.theta.append(np.zeros(shape=(self.neurons_per_layer[i+1], self.neurons_per_layer[i]+1)))

        with open(self.initial_weights_file) as f:
            data = f.readlines()
            layer_index = 0

            for line in data:
                line = line.strip('\n')
                weights = line.split(';')

                for neuronI in range(len(weights)):
                    # i = neurônio i da camada seguinte
                    i_weights = weights[neuronI].split(',')

                    for neuronJ in range(len(i_weights)):
                        # neuronJ = neurônio neuronJ da camada atual
                        self.theta[layer_index][neuronI][neuronJ] = float(i_weights[neuronJ])

                layer_index = layer_index + 1


    def initActivationsVector(self):
        # activation[L][j] = ativação/saída do neurônio j da camada L
        for i in range(len(self.neurons_per_layer)):
            # activation é uma lista de arrays Nx1, onde N=número de neurônios da
            # camada + 1 (bias)
            self.activation.append(np.zeros(shape=(self.neurons_per_layer[i]+1, 1)))


    def setDataset(self):
        fileUtils = FileUtils(self.dataset_file)
        self.dataset = fileUtils.getDataset()

    def setTrainingData(self):
        self.training_data = self.dataset

    def costFunction(self):
        ex1 = Instance(attributes=[0.13], classification=[0.9])
        ex2 = Instance(attributes=[0.42], classification=[0.23])
        training_data = [ex1, ex2]
        delta = []

        y = []
        f_theta = []
        J_vector = []

        J = 0 # acumula o erro total da rede

        for i in range(len(training_data)):
            f_theta.append(training_data[i].classification)
            y.append(self.backpropagation(training_data[i]))
            j_i = self.calculateError(f_theta[i], y[i])

            print('Saida predita para o exemplo {0} = {1}'.format(i, y[i]))
            print('Saida esperada para o exemplo {0} = {1}'.format(i, f_theta[i]))
            print('J do exemplo {0} = {1}'.format(i, j_i))

            # Calcula o erro da rede para cada instância i
            J_vector.append(j_i)

        # Calcula o erro da rede para todo o conjunto
        J = float(sum(J_vector)/len(training_data))


        # eleva cada peso da rede ao quadrado (exceto os pesos de bias) e os soma
        #S = (λ/(2 * num_examples)) * S

        # Retorna o custo regularizado J+S



    def backpropagation(self, instance):
        print('')
        # z é um vetor de tamanho num_layers
        z = [0 for i in range(self.num_layers)]
        bias = [1]

        print('--> Propagando entrada: {0}'.format(instance.attributes))

        self.activation[0] = bias + instance.attributes

        g_vector_function = np.vectorize(self.g)

        # Para cada camada k=2 (iniciando contagem em 1) até k=num_layers-1
        for k in range(1, self.num_layers-1):
            # Ativação dos neurônios da camada k
            # z[k] = self.theta[k-1] * self.activation[k-1]
            z[k] = np.dot(self.theta[k-1], self.activation[k-1])
            a = g_vector_function(z[k])
            self.activation[k] = np.insert(a, 0, 1, axis=0)


        # Ativação do neurônio da camada de saída
        k = self.num_layers-1

        z[k] = np.dot(self.theta[self.num_layers-2], self.activation[self.num_layers-2])

        self.activation[k] = g_vector_function(z[k])

        print('Vetor a = {0}'.format(self.activation))
        print('Vetor z = {0}'.format(z))
        print('Vetor de theta = {0}'.format(self.theta))
        print('Predicao final= {0}'.format(self.activation[k]))

        # Predição final
        return self.activation[k]


    def g(self, x):
        return float(1/(1 + math.exp(-x)))

    def calculateError(self, f_theta, y):
        # calcula o erro obtido a partir das saídas obtidas para um exemplo i
        matrix_f_theta = np.matrix(f_theta)
        matrix_y = np.matrix(y)

        # J(i) = -y(i) .* log(fθ(x(i))) - (1-y(i)) .* log(1 - fθ(x(i)))
        first_term = np.multiply(-matrix_y, np.log(matrix_f_theta))
        second_term = np.multiply((1 - matrix_y), np.log(1 - matrix_f_theta))
        J = first_term - second_term

        return J
