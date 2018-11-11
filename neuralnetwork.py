import math
import numpy as np

from instance import Instance
from utils import FileUtils

class NeuralNetwork(object):

    def __init__(self, initial_weights_file, neurons_per_layer, config_file, dataset):
        self.initial_weights_file = initial_weights_file

        self.output     = []
        self.prediction = []
        self.dataset = dataset
        self.num_layers = len(neurons_per_layer)
        self.neurons_per_layer = neurons_per_layer
        self.num_input = None
        self.num_hidden = None
        self.num_output = None
        self.reg_factor = 0.25

        self.training_data = None

        self.gradients_for_all_instances = []

        self.theta = []
        self.initTheta()

        self.activation = []
        self.initActivationsVector()

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


    def setTrainingData(self):
        self.training_data = self.dataset

    def backpropagation(self):
        # Inicializa vetor de gradientes
        regularized_gradients = []
        for g in range(len(self.neurons_per_layer) - 1):
            regularized_gradients.append(np.zeros(shape=(self.neurons_per_layer[g+1], self.neurons_per_layer[g]+1)))

        ex1 = Instance(attributes=[0.13], classification=[0.9])
        ex2 = Instance(attributes=[0.42], classification=[0.23])

        training_data = [ex1, ex2]

        delta = []
        y = []
        f_theta = []
        J_vector = []

        deltas = []
        for i in range(len(self.neurons_per_layer)-1):
            deltas.append(np.zeros(shape=(self.theta[i].shape)))

        last_layer_index = len(self.neurons_per_layer) - 1

        J = 0 # acumula o erro total da rede

        for i in range(len(training_data)):
            # Inicializa vetor de gradientes
            gradients = []
            for g in range(len(self.neurons_per_layer) - 1):
                gradients.append(np.zeros(shape=(self.neurons_per_layer[g+1], self.neurons_per_layer[g]+1)))

            # Propaga a instância e obtém as saídas preditas pela rede
            f_theta.append(training_data[i].classification)
            y.append(self.forwardPropagation(training_data[i]))

            # Calcula delta da camada de saída
            j_i = y[i] - f_theta[i]
            deltas.insert(last_layer_index, j_i)

            # Calcula delta das camadas ocultas
            for k in range(last_layer_index-1, 0, -1):
                #  [θ(l=k)]T 𝛿(l=k+1) .* a(l=k) .* (1-a(l=k))
                theta_copy = list(self.theta[k])
                # theta_matrix = np.delete(theta_copy, 0, axis=1) # remove o peso do neurônio de bias
                theta_matrix = np.matrix(theta_copy)

                theta_transp = theta_matrix.transpose()
                delta_matrix = np.matrix(deltas[k+1])
                first_term = theta_transp * delta_matrix

                activation_copy = np.array(self.activation[k])
                # activation_matrix = np.delete(activation_copy, 0, axis=0) # remove a ativação do neurônio de bias
                activation_matrix = np.matrix(activation_copy)
                second_term = np.multiply(first_term, activation_matrix)

                aux = 1 - activation_matrix
                third_term = np.multiply(second_term, aux)

                # remove o neurônio de bias
                # delta_k = np.delete(third_term, 0, 0)
                delta_k = third_term.diagonal() # gambiarra: alguma dimensão ficou errada, mas pega os valores certos
                deltas[k] = delta_k


            # Atualiza os gradientes dos pesos com base no exemplo atual
            # TODO: CALCULAR O DA ULTIMA CAMADA FORA DO FOR
            for k in range(last_layer_index-1, -1, -1):
                # D(l=k) = D(l=k) + 𝛿(l=k+1) [a(l=k)]T
                delta_matrix = np.matrix(deltas[k+1])
                activation_matrix = np.matrix(self.activation[k])

                if not k == last_layer_index-1:
                    # se não estiver calculando para a ultima camada, desconsidera o bias
                    delta_matrix = np.delete(delta_matrix, 0, 1).transpose()

                gradients[k] = gradients[k] + (delta_matrix * activation_matrix)


            # Guarda os gradientes calculados para a instância atual
            self.gradients_for_all_instances.append(gradients)

            print('Saida predita para o exemplo {0} = {1}'.format(i, y[i]))
            print('Saida esperada para o exemplo {0} = {1}'.format(i, f_theta[i]))
            print('Gradientes para o exemplo {0} = {1}'.format(i, gradients))

            print('J do exemplo {0} = {1}'.format(i, j_i))

            # Calcula o erro da rede para cada instância i
            J_vector.append(j_i)


        print('')
        print('Dataset completo processado. Calculando gradientes regularizados...')
        # Calcula gradientes finais (regularizados) para os pesos de cada camada
        # Inicializa vetor p
        p = []
        for g in range(len(self.neurons_per_layer) - 1):
            p.append(np.zeros(shape=(self.neurons_per_layer[g+1], self.neurons_per_layer[g]+1)))

        for k in range(last_layer_index-1, -1, -1):
            # p com a primeira coluna zerada // aplica regularização λ apenas a pesos não bias
            p[k] = np.multiply(self.reg_factor , self.theta[k])

            # combina gradientes com regularização; divide por #exemplos para calcular gradiente médio
            regularized_gradients[k] = 0


        # Calcula o erro da rede para todo o conjunto
        J = float(sum(J_vector)/len(training_data))

        # eleva cada peso da rede ao quadrado (exceto os pesos de bias) e os soma
        #S = (λ/(2 * num_examples)) * S

        # Retorna o custo regularizado J+S



    def forwardPropagation(self, instance):
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
            #z[k] = self.theta[k-1] * self.activation[k-1]
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
