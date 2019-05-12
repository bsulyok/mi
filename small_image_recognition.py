import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os
from itertools import tee
from tqdm import tqdm

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

class network:
    
    def __init__(self, hidden_layers):
        self.input_size = 784
        self.output_size = 10
        self.epoch = 0
        self.running_time = 0
        self.layers = []
        if hidden_layers[0] == 'new':
            if not hidden_layers[1]:
                layers_to_create = [400, 200]    
            else:
                layers_to_create = [int(l) for l in hidden_layers[1:]]
            self.layers = [np.random.randn(j, i+1)*np.sqrt(2/(i+1)) for i, j in pairwise([self.input_size] + layers_to_create + [self.output_size])]
        else:
            with open(hidden_layers, 'r') as f:
                layers_to_create, epoch, running_time = f.readlines()
                self.epoch = int(epoch[:-1])
                self.running_time = float(running_time)
            for ltc in layers_to_create[:-1].split(';'):
                self.layers.append(np.loadtxt('{}/{}.csv'.format(hidden_layers.split('.')[0], ltc), delimiter=';'))

    def train(self, X_train, Y_train, batch_size):
        start_time, stop_time = time.time(), 0
        for k in tqdm(range(int(len(X_train)/batch_size))):
            X, Y = X_train[k*batch_size:k*batch_size+batch_size], Y_train[k*batch_size:k*batch_size+batch_size]
            inputs = [np.hstack(img/255) for img in X[:batch_size]]
            targets = [one_hot(label) for label in Y[:batch_size]]
            sum_delta_vector = None
            for i, t in zip(inputs, targets):
                input_vector, output_vector = self.feedforward(i)
                delta_vector = self.backpropagate(input_vector, output_vector, t)
                sum_delta_vector = delta_add(sum_delta_vector, delta_vector)
            self.update_layers(sum_delta_vector, batch_size)
        self.epoch += 1
        self.running_time += (time.time()-start_time)
            #print('Finished batch #{} in {}!'.format(k, time.time()-stop_time))
            #stop_time = time.time()
        print('\nFinished epoch in {}'.format(time.time()-start_time))

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
        for k in range(len(self.layers)-1):
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

    def update_layers(self, delta_vector, batch_size):
        for layer_num, delta in zip(range(len(self.layers)), delta_vector):
            self.update_1(layer_num, delta, batch_size)

    def update_1(self, layer_num, c, batch_size):
        learning_rate = 1/(self.epoch + 10) 
        self.layers[layer_num] -= learning_rate*c/batch_size

    def guess(self, test_input):
        test_input = np.hstack(test_input/255)
        return self.feedforward(test_input)[1][-1]

    def get_error(self, output, target):
        a = output-target
        return sum((a)**2)/2

    def print_layers(self, filename):
        for layer in self.layers:
            np.savetxt('{}_{}.txt'.format(filename, len(layer)), layer, delimiter=';')

    def trained_guess(self, x_test):
        print(self.guess(x_test).argmax())
        plt.matshow(x_test)
        plt.show()
    
    def print_network(self, filename):
        out_string = ''
        for layer in self.layers:
            out_string+= '{};'.format(len(layer))
        out_string = out_string[:-1] + '\n{}\n{}'.format(self.epoch, self.running_time)
        with open('{}.csv'.format(filename), 'w') as f: f.write(out_string)
        if not os.path.exists(filename):
            os.mkdir(filename)
        for layer in self.layers:
            np.savetxt('{}/{}.csv'.format(filename,len(layer)), layer, delimiter=';')

    def full_test(self, x_test, y_test):
        correct = 0
        for sample in tqdm(range(len(x_test))):
            output = self.guess(x_test[sample])
            if output.argmax() == y_test[sample]:
                correct += 1
        return correct/len(x_test)


if sys.argv[1] == 'mnist':
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
elif sys.argv[1] == 'fashion_mnist':
    from keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
else:
    print('Dataset input not recognized. Use "mnist" or "fashion_mnist"!')

if sys.argv[2] == 'new':
    NN = network(sys.argv[2:])
else:
    NN = network(sys.argv[2])

#NN.train(x_train[1000:2000], y_train[1000:2000], 50)
#print('Finished in {} epochs'.format(NN.epoch))
#NN.print_network('train')
