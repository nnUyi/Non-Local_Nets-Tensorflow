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

def data_augement(image, is_training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.

    if is_training:
        # For training, add the following to the TensorFlow graph.

        # Randomly crop the input image.
        #size = image.get_shape().as_list()[0]
        #image = tf.random_crop(image, size=[size, img_size_cropped, img_size_cropped, num_channels])
        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        # For training, add the following to the TensorFlow graph.

        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        #image = tf.image.resize_image_with_crop_or_pad(image,
        #                                               target_height=img_size_cropped,
        #                                               target_width=img_size_cropped)
        pass
    return image

def pre_process(images, is_training):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    images = tf.map_fn(lambda image: data_augement(image, is_training), images)
    return images

if __name__ == '__main__':
    # test NonLocalNet
    input_x = tf.Variable(tf.random_normal([10,64,64,256]))
    out = NonLocalBlock(input_x, out_channels = 128)
    print(out.get_shape().as_list())
    # test data_augement
    input_x = tf.Variable(tf.random_normal([10,64,64,3]))
    image = pre_process(input_x, True)
    print(image.get_shape().as_list())
