import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import numpy as np

from tqdm import tqdm
from ops import NonLocalBlock, pre_process
from utils import get_data, gen_batch_data

class NonLocalNet:
    model_name = 'NonLocalNet.model'

    def __init__(self,
                config=None,
                sess=None,
                batchsize=32,
                input_height=28,
                input_width=28,
                input_channels=1,
                num_class=10):
        self.config = config
        self.batchsize =batchsize
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.num_class = num_class
        self.sess = sess

    def Net(self, input_x, is_training = True, scope='Nets'):
        batchsize, height, width, in_channels = input_x.get_shape().as_list()
        with tf.variable_scope(scope) as scope:
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn = None,
                                normalizer_fn = None,
                                weights_initializer = tf.truncated_normal_initializer(stddev=0.02),
                                weights_regularizer = None):
                with tf.name_scope('convolution') as sc_cnv:
                    # mnist: B*28*28*1
                    cnv1 = slim.conv2d(input_x, 32, [3,3], stride=1, scope='Net_cnv1', padding='SAME')
                    cnv1_bn = slim.batch_norm(cnv1, scope='Net_cnv1_bn')
                    cnv1_pool = tf.nn.relu(slim.max_pool2d(cnv1_bn, [2,2], stride=2, scope='Net_cnv1_pool'))
                    # mnist: B*14*14*1
                    nonlocal_block1 = NonLocalBlock(cnv1_pool, 32, scope='nonlocal_block1')
                    cnv2 = slim.conv2d(nonlocal_block1, 64, [3,3], stride=1, scope='Net_cnv2', padding='SAME')
                    cnv2_bn = slim.batch_norm(cnv2, scope='Net_cnv2_bn')
                    cnv2_pool = tf.nn.relu(slim.max_pool2d(cnv2_bn, [2,2], stride=2, scope='Net_cnv2_pool'))
                    # mnist: B*7*7*1
                    nonlocal_block2 = NonLocalBlock(cnv2_pool, 64, scope='nonlocal_block2')
                    cnv3 = slim.conv2d(nonlocal_block2, 128, [3,3], stride=1, scope='Net_cnv3', padding='SAME')
                    cnv3_bn = slim.batch_norm(cnv3, scope='Net_cnv3_bn')
                    cnv3_pool = tf.nn.relu(slim.max_pool2d(cnv3_bn, [2,2], stride=2, scope='Net_cnv3_pool'))
                    # mnist: B*4*4*1
                with tf.name_scope('fully_connected') as sc_fc:
                    cnv3_pool_flatten = tf.reshape(cnv3_pool, [batchsize, -1])
                    fc1 = tf.nn.relu(slim.fully_connected(cnv3_pool_flatten, 1024, scope='fc1'))
                    fc1_dropout = slim.dropout(fc1, 0.5)
                    fc2 = slim.fully_connected(fc1_dropout, 10, scope='fc2')
                    fc2_softmax = tf.nn.softmax(fc2, -1)
                    return fc2_softmax, fc2

    def build_model(self):
        # mnist size
        self.image_shape = [self.input_height*self.input_width*self.input_channels]
        self.label_shape = [self.num_class]
        # input images & labels
        self.input_images = tf.placeholder(tf.float32, [self.batchsize, self.input_height, self.input_width, self.input_channels], 'input_images')
        self.input_labels = tf.placeholder(tf.float32, [self.batchsize]+self.label_shape, 'input_labels')
        # data_augement if image is the colorful one
        if self.input_channels == 3:
            print('data_augement')
            self.input_augement_images = pre_process(self.input_images, self.config.is_training)
        else:
            self.input_augement_images = self.input_images
        # prediction
        pred_softmax, pred_logits = self.Net(self.input_augement_images)
        # loss function
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_logits, labels=self.input_labels))
        # AdamOptimizer
        self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate, beta1=self.config.beta1, beta2=self.config.beta2).minimize(self.loss)
        # accuracy rate
        self.accuracy_counter = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred_softmax,1), tf.argmax(self.input_labels,1)), tf.float32))
        self.accuracy = self.accuracy_counter/self.batchsize
        # add summary
        self.loss_summary = tf.summary.scalar('cross entropy loss', self.loss)
        self.accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
        self.summaries = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('./{}/{}'.format(self.config.log_dir, self.config.datasets), self.sess.graph)
        # save model
        self.saver = tf.train.Saver()

    def train_model(self):
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        is_load = self.load_model()
        if is_load:
            print('[***]load model successfully')
        else:
            print('[!!!]fail to load model')
        # load training data
        datasource = get_data(self.config.datasets)
        gen_data = gen_batch_data(datasource, self.batchsize)
        idxs = int(len(datasource.images)/self.batchsize)
        step = 0

        for epoch in range(self.config.epochs):
            counter = 0
            for idx in tqdm(range(idxs)):
                images, labels = next(gen_data)
                _, loss, summaries, train_counter = self.sess.run([self.optim,
                                                                    self.loss,
                                                                    self.summaries,
                                                                    self.accuracy_counter],
                                                                    feed_dict={
                                                                        self.input_images:images,
                                                                        self.input_labels:labels
                                                                    })
                counter = counter + train_counter
                step = step + 1
                self.summary_writer.add_summary(summaries, global_step=step)

            train_accuracy  = float(counter)/(idxs*self.batchsize)
            print('epoch[{}/{}]:training accuracy:{:.4f}'.format(epoch,self.config.epochs, train_accuracy))

            if np.mod(epoch, 5)==0:
                test_accuracy = self.test_model()
                print('epoch[{}/{}]:testing accuracy:{:.4f}'.format(epoch,self.config.epochs, test_accuracy))

            if np.mod(epoch, 10)==0:
                self.save_model()

    def test_model(self):
        if not self.config.is_training:
            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()
            is_load = self.load_model()
            if is_load:
                print('[***]load model successfully')
            else:
                print('[!!!]fail to load model')
                return
        datasource = get_data(self.config.datasets, is_training=False)
        gen_data = gen_batch_data(datasource, self.batchsize, is_training=False)
        ites = int(len(datasource.images)/self.batchsize)
        counter = 0
        for ite in range(ites):
            images, labels = next(gen_data)
            tmp = self.sess.run(self.accuracy_counter,
                                feed_dict={
                                    self.input_images:images,
                                    self.input_labels:labels
                                })
            counter = counter + tmp

        accuracy = float(counter)/float(ites*self.batchsize)
        return accuracy

    @property
    def model_dir(self):
        return './{}/{}'.format(self.config.checkpoint_dir,
                                   self.config.datasets)
    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        checkpoint = os.path.join(self.model_dir, self.model_name)
        self.saver.save(self.sess, checkpoint)

    def load_model(self):
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if not (ckpt and ckpt.model_checkpoint_path):
            return False
        checkpoint = os.path.join(self.model_dir, self.model_name)
        self.saver.restore(self.sess, checkpoint)
        return True

if __name__=='__main__':
    nonlocalnet = NonLocalNet()
    input_x = tf.Variable(tf.random_normal([2,28,28,1]))
    softmax = nonlocalnet.Net(input_x)
    softmax_sum = tf.reduce_sum(softmax, -1)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(softmax))
        print(sess.run(softmax_sum))
