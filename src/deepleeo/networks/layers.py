import tensorflow as tf

#TODO: This function was taken from GEE Hackaton. Review this and reimplement if necessary.
def conv_pool_layer(bottom, filters, params, kernel_size=3, training=True, name="", pool=True, pad="same"):
    with tf.variable_scope("Conv_layer_{}".format(name)):
        conv = tf.layers.conv2d(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            padding=pad,
            data_format='channels_last',
            activation=None,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(params['l2_reg_rate']),
            kernel_initializer=tf.initializers.variance_scaling(scale=params['var_scale_factor'], distribution="uniform"),
            name="convolution_{}".format(name)
        )
        norm = tf.layers.batch_normalization(inputs=conv, training=training)
        relu = tf.nn.relu(norm, name="relu_{}".format(name))

        if(pool):
            return tf.layers.max_pooling2d(
                relu,
                2,
                strides=2,
                padding="SAME",
                name="pool_{}".format(name))
        else:
            return relu

def up_conv_layer(bottom, num_filters, kernel_size, strides, params, out_size=None, name="", pad="valid"):
    with tf.variable_scope("UP_Conv_Layer_{}".format(name)):
        up_conv = tf.layers.conv2d_transpose(
            inputs=bottom,
            filters=num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=pad,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(params['l2_reg_rate']),
            kernel_initializer=tf.initializers.variance_scaling(scale=params['var_scale_factor'], distribution="uniform"),
            name="upconv{}".format(name)
        )

        if out_size is not None:
            upconv_shape = up_conv.shape

            if upconv_shape[1] != out_size:
                tf.cast(out_size, tf.int32)
                offsets = [0, tf.cast((int(upconv_shape[1]) - int(out_size)) / 2, tf.int32),
                           tf.cast((int(upconv_shape[2]) - int(out_size)) / 2, tf.int32), 0]
                size = [-1, out_size, out_size, num_filters]
                up_conv = tf.slice(up_conv, offsets, size)

        return up_conv

def up_conv_concat_layer(bottom, concat, params, kernel_size=4, num_filters=2, strides=2,
                         pad="valid", name=""):
    with tf.variable_scope("UP_Conv_Layer_{}".format(name)):
        upconv = tf.layers.conv2d_transpose(
            inputs=bottom,
            filters=num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=pad,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(params['l2_reg_rate']),
            kernel_initializer=tf.initializers.variance_scaling(scale=params['var_scale_factor'], distribution="uniform"),
            name="deconv_{}".format(name)
        )

        score_pool = tf.layers.conv2d(inputs=concat, filters=num_filters, kernel_size=1, padding="valid",
                                data_format="channels_last", activation=None, name="score_layer")

        upconv = tf.add(upconv, score_pool)

        return upconv

def resnet_base_layer(bottom, num_filters, strides):
    return False