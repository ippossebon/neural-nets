import math
import numpy as np

from instance import Instance
from utils import FileUtils

class NeuralNetwork(object):

    def __init__(self, initial_weights_file, neurons_per_layer, config_file, dataset):
        self.initial_weights_file = initial_weights_file

        self.dataset = dataset
        self.num_layers = len(neurons_per_layer)
        self.neurons_per_layer = neurons_per_layer
        self.reg_factor = 0.25
        self.learning_rate = 0.1
        self.epsilon = 0.000001

        self.training_data = None

        self.theta = []
        self.initTheta()

        self.setTrainingData()

        self.regularized_gradients = []

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
        # # Exemplo 1
        # ex1 = Instance(attributes=[0.13], classification=[0.9])
        # ex2 = Instance(attributes=[0.42], classification=[0.23])

        # Exemplo 2
        ex1 = Instance(attributes=[0.32, 0.68], classification=[0.75, 0.98])
        ex2 = Instance(attributes=[0.83, 0.02], classification=[0.75, 0.28])

        self.training_data = [ex1, ex2]

    def backpropagation(self):
        delta = []
        y = []
        f_theta = []
        J_vector = []
        deltas = []

        for i in range(len(self.neurons_per_layer)-1):
            deltas.append(np.zeros(shape=(self.theta[i].shape)))

        last_layer_index = len(self.neurons_per_layer) - 1

        J = 0 # acumula o erro total da rede

        # Inicializa vetor de gradientes
        D_matrix = []
        for g in range(len(self.neurons_per_layer) - 1):
            D_matrix.append(np.zeros(shape=(self.neurons_per_layer[g+1], self.neurons_per_layer[g]+1)))

        for i in range(len(self.training_data)):
            # Propaga a instância e obtém as saídas preditas pela rede
            y.append(self.training_data[i].classification)
            f, activation = self.forwardPropagation(self.training_data[i])
            f_theta.append(f)

            print('Saida predita para o exemplo {0} = {1}'.format(i, y[i]))
            print('Saida esperada para o exemplo {0} = {1}'.format(i, f_theta[i]))

            # Calcula delta da camada de saída
            j_i = f_theta[i] - y[i]
            deltas.insert(last_layer_index, j_i)

            # Calcula delta das camadas ocultas
            for k in range(last_layer_index-1, 0, -1):
                #  [θ(l=k)]T 𝛿(l=k+1) .* a(l=k) .* (1-a(l=k))
                theta_copy = list(self.theta[k])
                theta_matrix = np.matrix(theta_copy)

                theta_transp = theta_matrix.transpose()
                delta_matrix = np.asmatrix(deltas[k+1]).transpose()
                first_term = theta_transp * delta_matrix

                activation_copy = np.array(activation[k])
                activation_matrix = np.matrix(activation_copy)
                second_term = np.multiply(first_term, activation_matrix)

                aux = 1 - activation_matrix
                third_term = np.multiply(second_term, aux)

                delta_k = third_term.diagonal() # gambiarra: alguma dimensão ficou errada, mas pega os valores certos

                # remove o neurônio de bias
                delta_k_without_bias = np.delete(delta_k, 0, 1)
                deltas[k] = delta_k_without_bias


            for k in range(last_layer_index-1, -1, -1):
                # D(l=k) = D(l=k) + 𝛿(l=k+1) [a(l=k)]T
                delta_matrix = np.matrix(deltas[k+1])
                activation_matrix = np.matrix(activation[k])

                gradients = delta_matrix.transpose() * activation_matrix
                D_matrix[k] = D_matrix[k] + gradients

                print('Gradientes de theta {0} para o exemplo {1}'.format(k, i))
                print(gradients)

            # Calcula o custo
            j_i = self.cost(f_theta[i], y[i])
            J_vector.append(j_i)

            print('J do exemplo {0} = {1}'.format(i, j_i))

        print('')
        print('Dataset completo processado. Calculando gradientes regularizados...')
        # Calcula gradientes finais (regularizados) para os pesos de cada camada
        # Inicializa vetor p
        p_vector = []
        for g in range(len(self.neurons_per_layer) - 1):
            p_vector.append(np.zeros(shape=(self.neurons_per_layer[g+1], self.neurons_per_layer[g]+1)))

        for g in range(len(D_matrix)):
            self.regularized_gradients.append(np.zeros(shape=D_matrix[g].shape))

        for k in range(last_layer_index-1, -1, -1):
            # Seja P(l=k) igual à (λ .* θ(l=k)), mas com a primeira coluna zerada -> aplica regularização λ apenas a pesos não bias
            p_vector[k] = np.multiply(self.reg_factor, self.theta[k])

            # Zera a primeira coluna -> não devemos multiplicar os pesos de bias
            p_vector[k][:,0] = 0

            # combina gradientes com regularização; divide por #exemplos para calcular gradiente médio
            # D(l=k) = (1/n) (D(l=k) + P(l=k))
            self.regularized_gradients[k] = (1/len(self.training_data)) * (D_matrix[k] + p_vector[k])
            print('Gradientes finais de theta (com regularização) {0} para o exemplo {1}'.format(k, i))
            print(self.regularized_gradients[k])


        # Calcula custo total da rede
        J = float(sum(J_vector)/len(self.training_data))

        if self.reg_factor > 0:
            # Calcula S = eleva cada peso da rede ao quadrado (exceto os pesos de bias) e os soma
            s_factor = 0
            for k in range(last_layer_index-1, -1, -1):
                s_factor = s_factor + np.sum(np.power(self.theta[k][:, 1:], 2))

            s_factor = float((self.reg_factor/(2 * len(self.training_data)))) * s_factor

            # Calcula o custo regularizado
            J = J + s_factor

        print('J total do dataset = {0}'.format(J))

        # Atualiza pesos de cada camada com base nos gradientes
        for k in range(last_layer_index-1, -1, -1):
            # θ(l=k) = θ(l=k) - α .* D(l=k)
            self.theta[k] = np.asarray(np.multiply(self.theta[k], self.learning_rate * D_matrix[k]))

        return J

    def forwardPropagation(self, instance):
        print('')
        z = [0 for i in range(self.num_layers)]
        bias = [1]

        print('--> Propagando entrada: {0}'.format(instance.attributes))

        activation = []
        # activation[L][j] = ativação/saída do neurônio j da camada L
        for i in range(len(self.neurons_per_layer)):
            # activation é uma lista de arrays Nx1, onde N=número de neurônios da
            # camada + 1 (bias)
            activation.append(np.zeros(shape=(self.neurons_per_layer[i]+1, 1)))

        activation[0] = bias + instance.attributes

        g_vector_function = np.vectorize(self.g)

        # Para cada camada k=2 (iniciando contagem em 1) até k=num_layers-1
        for k in range(1, self.num_layers-1):
            # Ativação dos neurônios da camada k
            z[k] = np.dot(self.theta[k-1], activation[k-1])

            aux = g_vector_function(z[k])
            activation[k] = np.insert(aux, 0, 1, axis=0)

        # Ativação do neurônio da camada de saída
        k = self.num_layers-1

        z[k] = np.dot(self.theta[self.num_layers-2], activation[self.num_layers-2])

        activation[k] = g_vector_function(z[k])

        print('Vetor a =')
        print(activation)
        print('Vetor z =')
        print(z)
        print('Vetor de theta =')
        print(self.theta)
        print('Predicao final =')
        print(activation[k])

        # Predição final
        return activation[k], activation


    def g(self, x):
        return float(1/(1 + math.exp(-x)))


    def cost(self, f_theta, y):
        matrix_f_theta = np.asmatrix(f_theta)
        matrix_y = np.asmatrix(y)

        # J(i) = -y(i) .* log(fθ(x(i))) - (1-y(i)) .* log(1 - fθ(x(i)))
        first_term = np.multiply(-matrix_y, np.log(matrix_f_theta))
        second_term = np.multiply((1 - matrix_y), np.log(1 - matrix_f_theta))
        J = first_term - second_term
        J = np.sum(J)

        return J

    def verify(self):
        # Compara os valores obtidos pelo backpropagation com as derivadas reais
        print('')
        print('Rodando verificação numérica de gradientes (epsilon=0.0000010000)')

        y = []
        for i in range(len(self.training_data)):
            y.append(self.training_data[i].classification)

        for i in range(len(self.regularized_gradients)):
            print('Gradiente numérico de theta ', i)
            j1 = self.cost(self.regularized_gradients[i] + self.epsilon, y)
            j2 = self.cost(self.regularized_gradients[i] - self.epsilon, y)

            numeric_gradient = (j1 - j2)/ (2 * self.epsilon)
            print(numeric_gradient)


    def runNetwork(self, max_iter):
        j = 0
        new_j = 0

        for i in range(max_iter):
            j = new_j
            new_j = self.backpropagation()

            if abs(j - new_j) < self.epsilon:
                return self.theta
