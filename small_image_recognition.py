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

def relu(input_vector, relu_max):
    input_vector[input_vector < 0] = 0
    #input_vector[input_vector > relu_max] = relu_max
    return input_vector

def heaviside(input_vector):
    return input_vector >= 0

def softmax(input_vector):
    a = np.exp(input_vector - np.max(input_vector))
    return a/a.sum()

def softmax_deriv(input_vector):
    return np.diag(input_vector) - np.outer(input_vector, input_vector)

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
        my_dict = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    elif user_input == 'fashion_mnist':
        from keras.datasets import fashion_mnist as data
        my_dict = ['T-shirt', ' trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    else:
        print('Input dataset {} not recognized!\nTerminating'.format(user_input))
    return my_dict, data 

def variate(img):
    center = (img.shape[0]/2 + np.random.randint(-4, 5),img.shape[1]/2 + np.random.randint(-4, 5))
    angle = 60*np.random.random()-30
    scale = 0.4*np.random.random()+0.8
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, M, img.shape)

class network:
    
    def __init__(self, insize, netw, outsize, dataset):
        self.input_size = insize
        self.dataset = dataset
        self.output_size = outsize
        self.epoch = 0
        self.training_time = 0
        self.layers = []
        self.relu_max = 0
        self.network_name = 'new_network'
        if type(netw) == str and os.path.exists('{}'.format(netw)):
            self.network_name = netw
            with open('{}/data.csv'.format(self.network_name), 'r') as f:
                trained_layers, layers_to_create, epoch, training_time, relu_max = f.readlines()
                self.dataset = trained_layers[:-1]
                self.epoch = int(epoch[:-1])
                self.training_time = float(training_time[:-1])
                self.relu_max = float(relu_max)
            for ltc in layers_to_create[:-1].split(';'):
                self.layers.append(np.loadtxt('{}/{}.csv'.format(self.network_name, ltc), delimiter=';'))
        elif type(netw) == list:
            layers_to_create = [int(l) for l in netw]
            self.layers = [np.random.randn(j, i+1)*np.sqrt(2/(i+1)) for i, j in pairwise([self.input_size] + layers_to_create + [self.output_size])]
            if not os.path.exists('networks/{}'.format(self.network_name)):
                os.mkdir('networks/{}'.format(self.network_name))

    def train(self, train_data, batch_size, epoch=1, variation=0):
        ordering = list(range(len(train_data)))
        start_time = time.time()
        out_file = open('networks/{}/error.csv'.format(self.network_name), 'w')
        for ep in range(epoch):
            np.random.shuffle(ordering)
            shuffled_train_data = [train_data[p] for p in ordering]
            self.epoch_train(shuffled_train_data, batch_size, out_file, variation)
            if epoch>1:
                print('Finished epoch #{}!'.format(ep+1))
            self.epoch += 1
        out_file.close()
        self.training_time += time.time()-start_time
        print('Finished all {} epoch in {} seconds!'.format(epoch, time.time()-start_time))

    def epoch_train(self, epoch_train_data, batch_size, out_file, variation):
        batch_num = len(epoch_train_data)/batch_size
        for i in tqdm(range(0, len(epoch_train_data), batch_size)):
            mini_batch = epoch_train_data[i:i+batch_size]
            sum_delta_vector, sum_error = None, 0
            for epoch_train_datapoint in mini_batch:
                train_img = epoch_train_datapoint[0]
                if variation:
                    train_img = variate(train_img)
                train_input = train_img.flatten()/255
                train_target = one_hot(epoch_train_datapoint[1])
                input_vector, output_vector, final_output = self.feedforward(train_input)
                sum_error += self.get_error(final_output, train_target)
                c_vector = self.backpropagate(output_vector, final_output, train_target)
                delta_vector = []
                for c, inpt in zip(c_vector[::-1], input_vector):
                    delta = np.outer(c, append_1(inpt))
                    delta_vector.append(delta)
                sum_delta_vector = delta_add(sum_delta_vector, delta_vector)
            self.update_layers(sum_delta_vector, batch_size)
            out_file.write('{}\n'.format(sum_error))

    def feedforward(self, inpt, testing=0):
        output_vector = []
        input_vector = []
        for layer in self.layers:
            input_vector.append(inpt)
            outpt, act_outp = self.feed_1(layer, inpt, testing)
            inpt = act_outp
            output_vector.append(outpt)
        final_output = softmax(output_vector[-1])
        return input_vector, output_vector, final_output

    def feed_1(self, layer, a, testing=0):
        output = np.dot(layer, append_1(a))
        if not testing:
            self.relu_max = max(self.relu_max, max(output))
        activated_output = relu(output, self.relu_max)
        return output, activated_output

    def backpropagate(self, output_vector, final_output, target):
        c_vector = [np.dot(softmax_deriv(final_output), (final_output-target))]
        for k in range(len(self.layers)-1):
            c = self.backprop_1(self.layers[-k-1],output_vector[-k-2], c_vector[-1])
            c_vector.append(c)
        return c_vector

    def backprop_1(self, layer, outpt, delta):
        a = np.dot(layer.transpose(),delta)
        b = heaviside(outpt)
        return a[:-1]*b

    def update_layers(self, delta_vector, batch_size):
        for layer_num, delta in zip(range(len(self.layers)), delta_vector):
            self.update_1(layer_num, delta, batch_size)

    def update_1(self, layer_num, c, batch_size):
        learning_rate = 1E-2 
        self.layers[layer_num] -= learning_rate*c/batch_size

    def test(self, test_data, variation=0):
        correct = 0
        for sample in tqdm(test_data):
            test_img = sample[0]
            if variation:
                test_img = variate(test_img)
            sample_prediction = self.trained_guess(sample[0]).argmax()
            correct += sample_prediction == sample[1]
        return correct/len(test_data)
    
    def trained_guess(self, test_input):
        test_input = test_input.flatten()/255
        return self.feedforward(test_input)[1][-1]

    def guess(self, testing_datapoint, variation=0):
        test_img = testing_datapoint[0]
        if variation:
            test_img = variate(test_img)
        plt.imshow(test_img)
        plt.title('This is a {}.'.format(train_dict[self.trained_guess(test_img).argmax()]))
        plt.show()
    
    def get_error(self, output, target):
        a = output-target
        return sum((a)**2)/2

    def print_network(self, filename):
        out_string = '{}\n'.format(self.dataset)
        if not os.path.exists('{}'.format(filename)):
            os.mkdir('{}'.format(filename))
        for layer in self.layers:
            out_string+= '{};'.format(len(layer))
        out_string = out_string[:-1] + '\n{}\n{}\n{}'.format(self.epoch, self.training_time, self.relu_max)
        with open('{}/data.csv'.format(filename), 'w') as f: f.write(out_string)
        for layer in self.layers:
            np.savetxt('{}/{}.csv'.format(filename,len(layer)), layer, delimiter=';')

if len(sys.argv) == 1 or sys.argv[1] not in ['mnist', 'fashion_mnist']:
    print('Input dataset not recognized. Use "mnist" or "fashion_mnist"!')

train_dict, data = initialize(sys.argv[1])

(x_train, y_train), (x_test, y_test) = data.load_data()
input_size = len(x_test[0].flatten())
output_size = int(max(y_test))+1
network_to_create = None
if sys.argv[2] == 'new':
    network_to_create = [int(lay) for lay in sys.argv[3:]]
else:
    network_to_create = sys.argv[2]

training_data = [[x, y] for x, y in zip(x_train, y_train)]
testing_data = [[x, y] for x, y in zip(x_test, y_test)]

NN = network(input_size, network_to_create, output_size, sys.argv[1])

#NN.train(training_data, 50[, 1[, 0]])
#NN.guess(testing_data[0][, 0])
#NN.print_network('train')
#NN.test(testing_data[, 0])

def deceive(img, false_label):
    input_img = img[0].flatten()/255
    correct_in, correct_out, correct_final = NN.feedforward(input_img)
    differential_label = one_hot(false_label) - correct_final
    c_vector = NN.backpropagate(correct_out, correct_final, differential_label)
    differential_input = np.dot(NN.layers[0].transpose(), c_vector[-1])
    differential_image = 255*np.reshape(differential_input[:-1], (28,28))
    return differential_image
