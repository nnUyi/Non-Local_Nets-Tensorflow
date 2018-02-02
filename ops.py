import tensorflow as tf
import tensorflow.contrib.slim as slim

def NonLocalBlock(input_x, out_channels, sub_sample=True, is_bn=True, scope='NonLocalBlock'):
    batchsize, height, width, in_channels = input_x.get_shape().as_list()
    with tf.variable_scope(scope) as sc:
        with tf.variable_scope('g') as scope:
            g = slim.conv2d(input_x, out_channels, [1,1], stride=1, scope='g')
            if sub_sample:
                g = slim.max_pool2d(g, [2,2], stride=2, scope='g_max_pool')

        with tf.variable_scope('phi') as scope:
            phi = slim.conv2d(input_x, out_channels, [1,1], stride=1, scope='phi')
            if sub_sample:
                phi = slim.max_pool2d(phi, [2,2], stride=2, scope='phi_max_pool')

        with tf.variable_scope('theta') as scope:
            theta = slim.conv2d(input_x, out_channels, [1,1], stride=1, scope='theta')

        g_x = tf.reshape(g, [batchsize,out_channels, -1])
        g_x = tf.transpose(g_x, [0,2,1])

        theta_x = tf.reshape(theta, [batchsize, out_channels, -1])
        theta_x = tf.transpose(theta_x, [0,2,1])
        phi_x = tf.reshape(phi, [batchsize, out_channels, -1])

        f = tf.matmul(theta_x, phi_x)
        # ???
        f_softmax = tf.nn.softmax(f, -1)
        y = tf.matmul(f_softmax, g_x)
        y = tf.reshape(y, [batchsize, height, width, out_channels])
        with tf.variable_scope('w') as scope:
            w_y = slim.conv2d(y, in_channels, [1,1], stride=1, scope='w')
            if is_bn:
                w_y = slim.batch_norm(w_y)
        z = input_x + w_y
        return z

if __name__ == '__main__':
    input_x = tf.Variable(tf.random_normal([10,64,64,256]))
    out = NonLocalBlock(input_x, out_channels = 128)
    print(out.get_shape().as_list())
