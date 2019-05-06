import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from itertools import tee
import sys

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def append_1(array):
    return np.append(array, 1)

def one_hot(digit):
    a = np.zeros(10)
    a[digit] = 1
    return a

def relu(input_vector):
    input_vector[input_vector < 0] = 0
    return input_vector

def heaviside(input_vector):
    return input_vector >= 0

def softmax(input_vector):
    a = np.exp(input_vector - np.max(input_vector))
    return a/a.sum()

def softmax_deriv(input_vector):
    a = -np.outer(input_vector, input_vector)
    b = np.zeros((len(input_vector),len(input_vector)))
    np.fill_diagonal(b, input_vector)
    return a+b

def delta_add(a, b):
    if a == None:
        return b
    else:
        v_sum = []
        for v1, v2 in zip(a, b):
            v_sum.append(v1+v2)
        return v_sum

class neural_network:
    
    def __init__(self, hidden_layers):
        self.input_size = 784
        self.output_size = 10
        self.num_hidden = len(hidden_layers)
        self.iteration = 0
        self.layers = [np.random.randn(j, i+1)*np.sqrt(2/(i+1)) for i, j in pairwise([self.input_size] + hidden_layers + [self.output_size])]
        self.accuracy = 0
        self.layer_num = len(self.layers)

    def train(self, X_train, Y_train, X_test, Y_test, b, test_size):
        for k in range(int(len(X_train)/b)):
            X, Y = X_train[k*b:k*b+b], Y_train[k*b:k*b+b]
            inputs = [np.hstack(img/255) for img in X[:b]]
            targets = [one_hot(label) for label in Y[:b]]
            same = 0
            for p in range(20):
            #while same < 5:
                sum_delta_vector = None
                for i, t in zip(inputs, targets):
                    input_vector, output_vector = self.feedforward(i)
                    delta_vector = self.backpropagate(input_vector, output_vector, t)
                    sum_delta_vector = delta_add(sum_delta_vector, delta_vector)
                self.update_layers(sum_delta_vector)
                acc, err = self.acc_test(X_test, Y_test, b, test_size)
                self.iteration += 1
                if acc > self.accuracy:
                    self.accuracy = acc
                elif acc == self.accuracy:
                    same +=1
                print('Iteration #{}'.format(self.iteration))
            print('Reached accuracy of {} on batch #{}!'.format(self.accuracy, k))
    
    def feedforward(self, inpt):
        output_vector = []
        input_vector = []
        for layer in self.layers:
            input_vector.append(inpt)
            outpt, act_outp = self.feed_1(layer, inpt)
            inpt = act_outp
            output_vector.append(outpt)
        return input_vector, output_vector

    def feed_1(self, layer, a):
        output = np.dot(layer, append_1(a))
        activated_output = relu(output)
        return output, activated_output

    def backpropagate(self, input_vector, output_vector, target):
        c_vector = [output_vector[-1]-target]
        for k in range(self.layer_num-1):
            c = self.backprop_1(self.layers[-k-1],output_vector[-k-2], c_vector[-1])
            c_vector.append(c)
        delta_vector = []
        for c, inpt in zip(c_vector[::-1], input_vector):
            delta = np.outer(c, append_1(inpt))
            delta_vector.append(delta)
        return delta_vector

    def backprop_1(self, layer, outpt, delta):
        a = np.dot(layer.transpose(),delta)
        b = heaviside(outpt)
        return a[:-1]*b

    def update_layers(self, delta_vector):
        for layer_num, delta in zip(range(len(self.layers)), delta_vector):
            self.update_1(layer_num, delta)

    def update_1(self, layer_num, c):
        learning_rate = 1/(self.iteration + 10) 
        self.layers[layer_num] -= learning_rate*c/batch_size

    def guess(self, test_input):
        test_input = np.hstack(test_input/255)
        return self.feedforward(test_input)[1][-1]

    def get_error(self, output, target):
        a = output-target
        return sum((a)**2)/2

    def acc_test(self, X, Y, b, test_size):
        correct, err = 0, 0
        samples = np.random.randint(len(X), size=test_size)
        for sample in samples:
            output = self.guess(X[sample])
            correct += output.argmax() == Y[sample]
            err += self.get_error(output, Y[sample])
        return correct/test_size, err/test_size

    def print_network(self, filename):
        for layer in self.layers:
            np.savetxt('{}_{}.txt'.format(filename, len(layer)), layer, delimiter=';')

    def print_layer(self, layer):
        print(self.layers[layer])

    def trained_guess(self, x_test):
        print(self.guess(x_test).argmax())
        plt.matshow(x_test)
        plt.show()

parser = sys.argv
netw = [int(l) for l in parser[1:]]
if not netw:
    netw = [400, 200]

(x_train, y_train), (x_test, y_test) = mnist.load_data()

meta = 50
batch_size = 50
test_size = 20

NN = neural_network(netw)
NN.train(x_train[:meta], y_train[:meta], x_train[:5000], y_train[:5000], int(batch_size), int(test_size))
print('Finished in {} iterations'.format(NN.iteration))

#NN.print_network('train') 
