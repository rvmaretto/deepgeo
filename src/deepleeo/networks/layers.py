import tensorflow as tf

#TODO: This function was taken from GEE Hackaton. Review this and reimplement if necessary.
def conv_pool_layer(inputs, filters, kernel_size=3, training=True, name="", pool=True, pad="same"):
    with tf.variable_scope("Layer {}".format(name)):
        conv = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            padding=pad,
            data_format='channels_last',
            activation=None)
        norm = tf.layers.batch_normalization(inputs=conv, training=training)
        relu = tf.nn.relu(norm, name="relu_{}".format(name))

        if(pool):
            return tf.layers.max_pooling2d(relu, 2, stride=2, name="pool_{}".format(name))
        else:
            return relu

def up_conv_layer(inputs, filters, kernel_size, strides, name):
    deconv = tf.layers.conv2d_transpose(
        inputs=input,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        kernel_initializer=tf.initializers.variance_scaling(scale=0.001, distribution="normal"),
        name="upconv{}".format(name)
    )
    return deconv