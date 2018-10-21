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
        self.training_data = None

        # thetha[L][i][j]: L camada, i neurônio da camada seguinte, j neurônio camada atual
        # thetha[L][i][j] é o peso conectando o neurônio j da camada L ao neurônio i da camada seguinte
        max_neurons_per_layer = max(self.neurons_per_layer) + 1
        self.theta = [[[0 for l in range(self.num_layers)] for i in range(max_neurons_per_layer)] for j in range(max_neurons_per_layer)]
        self.initTheta()

        # activation[L][j] = ativação/saída do neurônio j da camada L
        self.activation = [[0 for l in range(self.num_layers)] for j in range(max_neurons_per_layer)]

        self.setDataset()
        self.setTrainingData()

    def initTheta(self):
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


    def setDataset(self):
        fileUtils = FileUtils(self.dataset_file)
        self.dataset = fileUtils.getDataset()

    def setTrainingData(self):
        self.training_data = self.dataset

    def g(self, x):
        return float(1/(1 + math.exp(-x)))


    def backpropagateData(self):
        ex1 = Instance(attributes=[0.13], classification=[0.9])
        ex2 = Instance(attributes=[0.42], classification=[0.23])
        training_data = [ex1, ex2]
        delta = []

        for inst in training_data:
            pred = self.backpropagation(inst)


    def backpropagation(self, instance):
        print('--> backpropagation para instancia: (x = {0}, y = {1})'.format(instance.attributes, instance.classification))
        # z é um vetor de tamanho num_layers
        z = [0 for i in range(self.num_layers)]
        bias = [1]

        self.activation[0] = bias + instance.attributes

        # Para cada camada k=2 (iniciando contagem em 1) até k=num_layers-1
        for k in range(1, self.num_layers-2):
            # Ativação dos neurônios da camada k
            # z[k] = self.theta[k-1] * self.activation[k-1]
            z[k] = np.dot(self.theta[k-1], self.activation[k-1])
            a = self.g(z[k])
            self.activation[k] = bias + a

        # Ativação do neurônio da camada de saída
        k = self.num_layers-1

        z[k] = np.dot(self.theta[self.num_layers-2], self.activation[self.num_layers-2])

        g_vector_function = np.vectorize(self.g)
        self.activation[k] = g_vector_function(z[k])

        print('Vetor a = {0}'.format(self.activation))
        print('Vetor z = {0}'.format(z))
        print('Vetor de theta = {0}'.format(self.theta))
        print('Predicao para k={0} = {1}'.format(k, self.activation[k]))

        # Predição final
        return self.activation[k]


    def calculateError(self, f_x, y):
        # J(i) = -y(i) .* log(fθ(x(i))) - (1-y(i)) .* log(1 - fθ(x(i)))
        error = float(-y )


    def networkCostFunction(self):
        J = 0 # J acumula o erro total da rede

        for i in range(len(training_data)):
            J = self.costFunction(training_data[i])

        # retorna o custo total da rede
        return J


    def costFunction(self, instance, J, num_examples):
        prediction = self.backpropagation(instance)

        # Calcula o vetor J(i) com o custo associado à cada saída da rede
        # para o exemplo atual
        prediction_J = []
        for i in range(len(prediction)):
            prediction_J[i] = self.calculateError(f_x, instance.classification)
            J  = J + sum(prediction_J)

            # Divide o erro total calculado pelo número de exemplos
            J = float(J/num_examples)

            # eleva cada peso da rede ao quadrado (exceto os pesos de bias) e os soma
            S = (λ/(2 * num_examples)) * S

        # Retorna o custo regularizado J+S
        return J+S


    # TODO: executa esse processo até o criterio de parada -- definir critério
    def mainAlgorithm(self):
        ex1 = Instance(attributes=[0.13], classification=[0.9])
        ex2 = Instance(attributes=[0.42], classification=[0.23])
        training_data = [ex1, ex2]
        delta = []

        for inst in training_data:
            f_x = self.backpropagation(inst)
            error = float(f_x - inst.classification)
            delta.append(error)

            # calcula os deltas para as camadas ocultas
            for k in range(1, self.num_layers-1):
                # Remove o primeiro elemento de 𝛿(l=k) (i.e., o delta associado ao neurônio de bias da camada k)
                # 𝛿(l=k) = [θ(l=k)]T 𝛿(l=k+1) .* a(l=k) .* (1-a(l=k))
                pass
            # Para cada camada k=L-1...1
            for k in range(self.num_layers-1, 0):
                # atualiza os gradientes dos pesos de cada camada com base no exemplo atual
                # acumula em D(l=k) os gradientes com base no exemplo atual
                # D(l=k) = D(l=k) + 𝛿(l=k+1) [a(l=k)]T
                pass

        # Calcula gradientes finais (regularizados) para os pesos de cada camada
        for k in range(self.num_layers-1, 0):
            # Seja P(l=k) igual à (λ .* θ(l=k)), mas com a primeira coluna zerada // aplica regularização λ apenas a pesos não bias
            # D(l=k) = (1/n) (D(l=k) + P(l=k)) // combina gradientes com regularização; divide por #exemplos para calcular gradiente médio
            pass
        #3. // Nesse ponto, D(l=1) contém os gradientes dos pesos em θ(l=1); ...; D(l=L-1) contém os gradientes dos pesos em θ(l=L-1)

        # atualiza pesos de cada camada com base nos gradientes
        for k in range(self.num_layers-1, 0):
            # θ(l=k) = θ(l=k) - α .* D(l=k)
            pass
