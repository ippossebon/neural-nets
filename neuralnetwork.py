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
        self.reg_factor = 0
        self.learning_rate = 0.1

        self.training_data = None

        self.theta = []
        self.initTheta()

        self.activation = []
        self.initActivationsVector()

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
        ex1 = Instance(attributes=[0.32, 0.68], classification=[0.75, 0.98])
        ex2 = Instance(attributes=[0.83, 0.02], classification=[0.75, 0.28])

        self.training_data = [ex1, ex2]
        # self.training_data = self.dataset

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
            f_theta.append(self.training_data[i].classification)
            y.append(self.forwardPropagation(self.training_data[i]))

            print('Saida predita para o exemplo {0} = {1}'.format(i, y[i]))
            print('Saida esperada para o exemplo {0} = {1}'.format(i, f_theta[i]))

            # Calcula delta da camada de saída
            j_i = y[i] - f_theta[i]
            deltas.insert(last_layer_index, j_i)

            # Calcula delta das camadas ocultas
            for k in range(last_layer_index-1, 0, -1):
                import ipdb; ipdb.set_trace()
                #  [θ(l=k)]T 𝛿(l=k+1) .* a(l=k) .* (1-a(l=k))
                theta_copy = list(self.theta[k])
                # theta_matrix = np.delete(theta_copy, 0, axis=1) # remove o peso do neurônio de bias
                theta_matrix = np.matrix(theta_copy)

                theta_transp = theta_matrix.transpose()
                delta_matrix = np.asmatrix(deltas[k+1]).transpose() # TODO: TIVE QUE FAZER P conjunto 2
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

                # TODO: está alterando a forma da matrix de deltas
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

                gradients = delta_matrix * activation_matrix
                D_matrix[k] = D_matrix[k] + gradients

                print('Gradientes de theta {0} para o exemplo {1} = {2}'.format(k, i, gradients))


            print('J do exemplo {0} = {1}'.format(i, j_i))

            # Calcula o erro da rede para cada instância i
            J_vector.append(j_i)


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
            print('Gradientes finais de theta (com regularização) {0} para o exemplo {1} = {2}'.format(k, i, self.regularized_gradients[k]))


        # Atualiza pesos de cada camada com base nos gradientes
        for k in range(last_layer_index-1, -1, -1):
            # θ(l=k) = θ(l=k) - α .* D(l=k)
            self.theta[k] = np.multiply(self.theta[k], self.learning_rate * D_matrix[k])

        # Calcula o erro da rede para todo o conjunto
        J = float(sum(J_vector)/len(self.training_data))

        # Calcula S = eleva cada peso da rede ao quadrado (exceto os pesos de bias) e os soma
        s_factor = 0
        for k in range(last_layer_index-1, -1, -1):
            s_factor = s_factor + np.sum(np.power(self.theta[k][:, 1:], 2))

        s_factor = float((self.reg_factor/(2 * len(self.training_data)))) * s_factor

        # Calcula o custo regularizado
        J = J + s_factor


    def forwardPropagation(self, instance):
        print('')
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

    def verify(self):
        print('')
        print('Rodando verificacao numerica de gradientes (epsilon=0.0000010000)')

        for i in range(len(self.regularized_gradients)):
            print('Gradiente numérico de theta ', i)
            print(self.regularized_gradients[i])
