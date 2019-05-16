import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os
import cv2
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

def initialize(user_input):
    my_dict, data = None, None
    if user_input == 'mnist':
        from keras.datasets import mnist as data
        my_dict =  {0:' 0', 1:' 1', 2:' 2', 3:' 3', 4:' 4', 5:' 5', 6:' 6', 7:' 7', 8:'n 8', 9:' 9'}
    elif user_input == 'fashion_mnist':
        from keras.datasets import fashion_mnist as data
        my_dict = {0:' T-shirt', 1:' trouser', 2:' pullover', 3:' dress', 4:' coat', 5:' sandal', 6:' shirt', 7:' sneaker', 8:' bag', 9:'n ankle boot'}
    elif user_input == 'cifar10':
        from keras.datasets import cifar10 as data
        my_dict = {0:'n airplane', 1:'n automobile', 2:' bird', 3:' cat', 4:' deer', 5:' dog', 6:' frog', 7:' horse', 8:' ship', 9:' truck'}
    elif user_input == 'cifar100':
        from keras.datasets import cifar100 as data
        my_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9}
    else:
        print('Dataset {} not understood!\nTerminating'.format(user_input))
        quit()
    return my_dict, data 

class network:
    
    def __init__(self, insize, netw, outsize, dataset):
        self.input_size = insize
        self.dataset = dataset
        self.output_size = outsize
        self.epoch = 0
        self.running_time = 0
        self.layers = []
        if type(netw) is list:
            layers_to_create = [int(l) for l in netw]
            self.layers = [np.random.randn(j, i+1)*np.sqrt(2/(i+1)) for i, j in pairwise([self.input_size] + layers_to_create + [self.output_size])]
        elif type(netw) is str:
            with open('networks/{}/data.csv'.format(netw), 'r') as f:
                trained_dataset, layers_to_create, epoch, running_time = f.readlines()
                if trained_dataset != self.dataset[:-1]:
                    print('Datasets not alligned! This network has been trained for {} dataset, but called for usage on {}.\nTerminating process!'.format(trained_dataset[:-1], self.dataset))
                    quit()
                self.dataset = trained_dataset[:-1]
                self.epoch = int(epoch[:-1])
                self.running_time = float(running_time)
            for ltc in layers_to_create[:-1].split(';'):
                self.layers.append(np.loadtxt('networks/{}/{}.csv'.format(netw.split('.')[0], ltc), delimiter=';'))

    def train(self, X_train, Y_train, batch_size):
        start_time, stop_time = time.time(), 0
        for k in tqdm(range(int(len(X_train)/batch_size))):
            X, Y = X_train[k*batch_size:k*batch_size+batch_size], Y_train[k*batch_size:k*batch_size+batch_size]
            inputs = [intensity.flatten()/255 for intensity in X]
            targets = [one_hot(label) for label in Y]
            print(targets)
            sum_delta_vector = None
            for i, t in zip(inputs, targets):
                input_vector, output_vector = self.feedforward(i)
                delta_vector = self.backpropagate(input_vector, output_vector, t)
                sum_delta_vector = delta_add(sum_delta_vector, delta_vector)
            self.update_layers(sum_delta_vector, batch_size)
        self.epoch += 1
        self.running_time += (time.time()-start_time)
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

    def trained_guess(self, test_input):
        test_input = test_input.flatten()/255
        return self.feedforward(test_input)[1][-1]

    def get_error(self, output, target):
        a = output-target
        return sum((a)**2)/2

    def print_layers(self, filename):
        for layer in self.layers:
            np.savetxt('{}_{}.txt'.format(filename, len(layer)), layer, delimiter=';')

    def guess(self, x_test):
        plt.imshow(x_test)
        plt.title('This is a{}.'.format(train_dict[self.trained_guess(x_test).argmax()]))
        plt.show()
    
    def print_network(self, filename):
        out_string = '{}\n'.format(self.dataset)
        if not os.path.exists('networks/{}'.format(filename)):
            os.mkdir('networks/{}'.format(filename))
        for layer in self.layers:
            out_string+= '{};'.format(len(layer))
        out_string = out_string[:-1] + '\n{}\n{}'.format(self.epoch, self.running_time)
        with open('networks/{}/data.csv'.format(filename), 'w') as f: f.write(out_string)
        for layer in self.layers:
            np.savetxt('networks/{}/{}.csv'.format(filename,len(layer)), layer, delimiter=';')

    def test(self, x_test, y_test):
        correct = 0
        for sample in tqdm(range(len(x_test))):
            output = self.trained_guess(x_test[sample])
            if output.argmax() == y_test[sample]:
                correct += 1
        return correct/len(x_test)
    

if len(sys.argv) == 1 or sys.argv[1] not in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']:
    print('Dataset input not recognized. Use "mnist", "fashion_mnist", "cifar10" or "cifar100"!')
    quit()

train_dict, data = initialize(sys.argv[1])
(x_train, y_train), (x_test, y_test) = data.load_data()
input_size = len(x_test[0].flatten())
output_size = int(max(y_test))+1
NN = network(input_size, sys.argv[2:], output_size, sys.argv[1])

#x_train = [intensity.flatten()/255 for intensity in x_train]
#x_test = [intensity.flatten()/255 for intensity in x_test]

#NN.train(x_train[1000:2000], y_train[1000:2000], 50)
#print('Finished in {} epochs'.format(NN.epoch))
#NN.print_network('train')
