import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from itertools import tee

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def append_1(array):
    return np.append(array, 1)

def one_hot(digit):
    a = np.zeros((10))
    a[digit] = 1
    return a

def relu(input_vector):
    return np.maximum(0, input_vector)

def heaviside(input_vector):
    return input_vector != 0

def softmax(input_vector):
    a = np.exp(input_vector - np.max(input_vector))
    return a/a.sum()

def softmax_deriv(input_vector):
    a = -np.outer(input_vector, input_vector)
    b = np.zeros((len(input_vector),len(input_vector)))
    np.fill_diagonal(b, input_vector)
    return a+b

class neural_network:
    
    def __init__(self, hidden_layers):
        self.input_size = 784
        self.output_size = 10
        self.num_hidden = len(hidden_layers)
        self.iteration = 0
        self.layers = [np.random.randn(j, i+1)*np.sqrt(2/(i+1)) for i, j in pairwise([self.input_size] + hidden_layers + [self.output_size])]
        self.accuracy = 0

    def train(self, X_train, Y_train, X_test, Y_test, b):
        for k in range(int(len(X_train)/b)):
            X, Y = X_train[k*b:k*b+b], Y_train[k*b:k*b+b]
            inputs = [np.hstack(img/255) for img in X[:b]]
            targets = [one_hot(label) for label in Y[:b]]
            same = 0
            while same < 5:
                for i, j in zip(inputs, targets):
                    outputs, activated_outputs = self.feedforward(i)
                    self.backpropagate(outputs, activated_outputs, j)
                acc, err = self.acc_test(X_test, Y_test, b)
                self.iteration += 1
                if acc > self.accuracy:
                    self.accuracy = acc
                elif acc == self.accuracy:
                    same +=1
                print(acc)
            print('Reached accuracy of {} on batch #{}!'.format(self.accuracy, k))
    
    def backpropagate(self, output_vector, activated_vector, target):
        coeff = 1./(self.iteration + 30) 
        deltas = [output_vector[-1]-target]
        for k in range(len(activated_vector)):
            a = self.layers[-k-1].transpose().dot(deltas[-1])
            b = heaviside(output_vector[-k-2])
            deltas.append(a[:-1]*b)
        for k in range(len(activated_vector)):
            c = np.outer(deltas[k], append_1(output_vector[-k-2]))
            self.layers[-k-1] -= coeff*c

    def feedforward(self, input_vector):
        outputs, activated_outputs = [], []
        for layer in self.layers:
            output = np.dot(layer, append_1(input_vector))
            outputs.append(output)
            if not np.array_equal(layer, self.layers[-1]):
                activated_outputs.append(relu(output))
            input_vector = activated_outputs[-1]
        return outputs, activated_outputs
    
    def guess(self, test_input):
        return self.feedforward(test_input)[0][-1]

    def get_error(self, output, target):
        a = output-one_hot(target)
        return sum((a)**2)/784

    def acc_test(self, X, Y, b):
        correct, err = 0, 0
        samples = np.random.randint(len(X), size=b)
        for sample in samples:
            output = self.guess(X[sample])
            correct += output.argmax() == Y[sample]
            err += self.get_error(output, Y[sample])
        return correct/len(Y), err/len(Y)

    def print_network(self, filename):
        for layer in self.layers:
            np.savetxt('{}_{}'.format(filename, len(layer)), layer, delimiter=';')

    def print_layer(self, layer):
        print(self.layers[layer])

(x_train, y_train), (x_test, y_test) = mnist.load_data()


meta = 10
NN = neural_network([400, 200])
NN.train(x_train[:meta], y_train[:meta], x_test[:meta], y_test[:meta], int(meta))

print('Finished in {} iterations'.format(NN.iteration))

#NN.print_layer(0)

#print(NN.guess(x_train[0]))
#print(y_train[0])
#NN.print_network('trained.txt') 
