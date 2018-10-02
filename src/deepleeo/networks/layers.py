import tensorflow as tf

#TODO: This function was taken from GEE Hackaton. Review this and reimplement if necessary.
def conv_pool_layer(bottom, filters, kernel_size=3, training=True, name="", pool=True, pad="same"):
    with tf.variable_scope("Layer_{}".format(name)):
        conv = tf.layers.conv2d(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            padding=pad,
            data_format='channels_last',
            activation=None)
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

def up_conv_layer(bottom, filters, kernel_size, strides, name):
    deconv = tf.layers.conv2d_transpose(
        inputs=bottom,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        kernel_initializer=tf.initializers.variance_scaling(scale=0.001, distribution="normal"),
        name="upconv{}".format(name)
    )
    return deconv

#TODO: Put the concatenation here (tf.add)
#TODO: Verify what is wrong
#TODO: Put FCN32 in another file, creating the strategy "fcn32s"
def up_conv_concat_layer(bottom, concat, out_shape, kernel_size=4, num_filters=2, strides=2, pad="valid", name=""):
    # if pad == "valid":
    #     kernel_h = out_shape[1] - (2 * strides)
    #     kernel_w = out_shape[2] - (2 * strides)
    # elif pad == "same":
    #     kernel_h = ((bottom.shape[1] - 1) * strides) + 1
    #     kernel_w = ((bottom.shape[2] - 1) * strides) + 1
    
    with tf.variable_scope("UP_Layer_{}".format(name)):
        # print("SHAPE BOTTOM: ", bottom.get_shape())
        # print("SHAPE POOL: ", concat.get_shape())
        # filter_shape = [kernel_size, kernel_size, num_filters, bottom.get_shape()[3].value]
        # print("FILTER: ", filter_shape)
        # out_shape = [bottom.get_shape()[0].value, out_shape[1].value, out_shape[2].value, num_filters]
        # strides = [1, strides, strides, 1]
        # print("OUT_SHAPE: ", out_shape)
        # deconv = tf.nn.conv2d_transpose(
        #     bottom,
        #     filter_shape,
        #     out_shape,
        #     strides=strides,
        #     padding=pad,
        #     name=name
        # )

        # print("SHAPE DECONV: ", deconv.shape)
        deconv = tf.layers.conv2d_transpose(
            inputs=bottom,
            filters=num_filters,
            kernel_size=(kernel_size, kernel_size),
            strides=strides,
            padding=pad,
            kernel_initializer=tf.initializers.variance_scaling(scale=0.001, distribution="normal"),
            name="deconv_{}".format(name)
        )

        score_pool = tf.layers.conv2d(inputs=concat, filters=num_filters, kernel_size=1, padding="same",
                                data_format="channels_last", activation=None, name="score_layer")

        print("SHAPE SCORE_POOL: ", score_pool.shape)
        upconv = tf.add(deconv, score_pool)

        return upconv