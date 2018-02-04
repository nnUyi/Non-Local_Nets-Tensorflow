import tensorflow as tf
import os

from Non_Local_Nets import NonLocalNet

flags = tf.app.flags
flags.DEFINE_string('datasets', 'mnist', 'datasets type')
flags.DEFINE_bool('is_training', False, 'training phase')
flags.DEFINE_bool('is_testing', False, 'testing phase')
flags.DEFINE_integer('epochs', 100, 'training epochs')
flags.DEFINE_float('learning_rate', 0.0002, 'learning rate')
flags.DEFINE_float('beta1', 0.5, 'beta1')
flags.DEFINE_float('beta2', 0.999, 'beta2')
flags.DEFINE_integer('batchsize', 64, 'batchsize')
flags.DEFINE_integer('input_height', 28, 'input height')
flags.DEFINE_integer('input_width', 28, 'input width')
flags.DEFINE_integer('input_channels', 1, 'input channels')
flags.DEFINE_integer('num_class', 10, 'number of classes')
flags.DEFINE_string('checkpoint_dir', 'checkpoint', 'directory for saving model')
flags.DEFINE_string('data_dir', 'data', 'directory for storing training and testing data')
flags.DEFINE_string('log_dir', 'logs', 'directory for storing training logs')
FLAGS = flags.FLAGS

def check_dir():
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.mkdir(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)
    if not os.path.exists(FLAGS.data_dir):
        os.mkdir(FLAGS.data_dir)

def print_config():
    print('\n')
    print('ConfigProto')
    print('-'*30)
    print('datasets:{}'.format(FLAGS.datasets))
    print('is_training:{}'.format(FLAGS.is_training))
    print('is_testing:{}'.format(FLAGS.is_testing))
    print('epochs:{}'.format(FLAGS.epochs))
    print('learning_rate:{}'.format(FLAGS.learning_rate))
    print('beta1:{}'.format(FLAGS.beta1))
    print('beta2:{}'.format(FLAGS.beta2))
    print('height:{}'.format(FLAGS.input_height))
    print('width:{}'.format(FLAGS.input_width))
    print('channels:{}'.format(FLAGS.input_channels))
    print('num_class:{}'.format(FLAGS.num_class))
    print('-'*30)

def main(_):
    check_dir()
    print_config()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        nonlocalnet = NonLocalNet(config=FLAGS,
                                  sess=sess,
                                  batchsize=FLAGS.batchsize,
                                  input_height=FLAGS.input_height,
                                  input_width=FLAGS.input_width,
                                  input_channels=FLAGS.input_channels,
                                  num_class=FLAGS.num_class
                                  )
        nonlocalnet.build_model()
        if FLAGS.is_training:
            nonlocalnet.train_model()
        if FLAGS.is_testing:
            accuracy = nonlocalnet.test_model()
            print('testing accuracy:{:.4f}'.format(accuracy))

if __name__=='__main__':
    with tf.device('/gpu:0'):
        tf.app.run()
