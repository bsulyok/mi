import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from itertools import tee

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def one_hot(digit):
    hot = np.zeros((10))
    hot[digit] = 1
    return hot

def activate(output_vector):
    return np.asarray([1/(1+np.exp(-x)) for x in output_vector])

class neural_network:
    
    def __init__(self, hidden_layers):
        self.input_size = 784
        self.output_size = 10
        self.num_hidden = len(hidden_layers)
        self.iteration = 0
        self.layers = [np.random.rand(j, i+1) for i, j in pairwise([784] + hidden_layers + [10])]
        self.accuracy = 0

    def train(self, X, Y, X_test, Y_test, batch_size):
        inputs = [img.flatten() for img in X[:batch_size]]
        targets = [one_hot(label) for label in Y[:batch_size]]
        while self.accuracy < 9:
            for input_vector, target_vector in zip(inputs, targets):
                self.backpropagate(input_vector, target_vector)
            self.iteration += 1
            self.accuracy = self.acc_test(X_test, Y_test)

    def backpropagate(self, input_vector, target_vector):
        coeff = 1./(self.iteration + 10) 
        output_vector = self.feedforward(input_vector)
        y = output_vector[-1]
        deltas = [(target_vector-y)*y*(1-y)]
        for layer, output in zip(self.layers[-1:0:-1], output_vector[1::-1]):
            output = np.append(output, 1)
            deltas.append(((np.matrix.transpose(layer).dot(deltas[-1]))*output*(1-output))[:-1])
        for layer, output, delta in zip(self.layers[::-1], output_vector[-2::-1], deltas):
            k = np.outer(delta, np.append(output, 1))
            layer -= coeff*k

    def feedforward(self, input_vector):
        outputs = []
        for layer in self.layers:
            outputs.append(activate(layer.dot(np.append(input_vector, 1))))
            input_vector = outputs[-1] 
        return outputs
    
    def guess(self, test_input):
        return self.feedforward(test_input)[-1].argmax()

    def acc_test(self, X, Y):
        correct, k = 0, 0
        #samples = np.random.randint(5000, size=100)
        samples = np.arange(10)
        for sample in samples:
            correct += self.guess(X[sample]) == Y[sample]
            print(k)
            k += 1
        return correct

    def print_network(self, filename):
        for layer in self.layers:
            np.savetxt('{}_{}'.format(filename, len(layer)), layer, delimiter=';')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

NN = neural_network([400, 200])
NN.train(x_train[:10], y_train[:10], x_test[:10], y_test[:10], 5)

print('Finished in {} iterations'.format(NN.iteration))

NN.print_network('trained.txt') 
