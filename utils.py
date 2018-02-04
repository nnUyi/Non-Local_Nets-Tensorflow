import random
import numpy as np
import cifar10
from tensorflow.examples.tutorials.mnist import input_data

class Datasource:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

def get_data(data_type='mnist', is_training=True):
    if data_type == 'mnist':
        raw_data = input_data.read_data_sets('./data/mnist/', one_hot=True)
        shape = [28,28,1]
        if is_training:
            size = len(raw_data.train.images)
            images = np.reshape(raw_data.train.images, [size]+shape)
            labels = raw_data.train.labels
        else:
            size = len(raw_data.test.images)
            images = np.reshape(raw_data.test.images, [size]+shape)
            labels = raw_data.test.labels
    elif data_type == 'cifar10':
        if is_training:
            images, _, labels = cifar10.load_training_data()
        else:
            images, _, labels = cifar10.load_test_data()

    else:
        raise Exception('data type error: {}'.format(data_type))

    datasource = Datasource(images, labels)
    return datasource

def gen_data(datasource, is_training=True):
    while True:
        indices = range(len(datasource.images))
        random.shuffle(indices)
        if is_training:
            pass
        for i in indices:
            image = datasource.images[i]
            label = datasource.labels[i]
            yield image, label

def gen_batch_data(datasource, batchsize, is_training=True):
    data_gen = gen_data(datasource, is_training=is_training)
    while True:
        images = []
        labels = []
        for i in range(batchsize):
            image, label = next(data_gen)
            images.append(image)
            labels.append(label)
        yield np.array(images), np.array(labels)

# test
if __name__=='__main__':
    mnist = input_data.read_data_sets("./mnist/", one_hot=True)
    datasource = get_data(mnist)
    data_gen = gen_batch_data(datasource, 10)
    for i in range(10):
        images, labels = next(data_gen)
        print(images.shape)
        print(labels.shape)
