import random
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class Datasource:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

def get_data(mnist, is_training=True):
    if is_training:
        images = mnist.train.images
        labels = mnist.train.labels
    else:
        images = mnist.test.images
        labels = mnist.test.labels
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
