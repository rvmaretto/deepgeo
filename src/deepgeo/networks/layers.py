import tensorflow as tf


# TODO: Refactor. Re-implement this method in the following structure:
# conv_pool_layer(bottom, filters=[F1,F2,..., FN], poolings=[1,2,...,n], kernel_sizes=[k1, k2, ..., kn])
def conv_pool_layer(bottom, filters, params, kernel_size=3, training=True, name='', pool=True, pad='same'):
    with tf.compat.v1.variable_scope('Conv_layer_{}'.format(name)):
        conv = tf.compat.v1.layers.conv2d(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            padding=pad,
            data_format='channels_last',
            activation=None,
            kernel_regularizer=tf.keras.regularizers.L2(params['l2_reg_rate']),
            kernel_initializer=tf.keras.initializers.GlorotUniform(),  # tf.initializers.variance_scaling(scale=params['var_scale_factor'], distribution='uniform'),
            name='convolution_{}'.format(name)
        )
        norm = tf.compat.v1.layers.batch_normalization(inputs=conv, training=training, name='batch_norm_{}'.format(name))
        relu = tf.nn.relu(norm, name='relu_{}'.format(name))

        if pool:
            pooling = tf.compat.v1.layers.max_pooling2d(
                relu,
                2,
                strides=2,
                padding=pad,
                name='pool_{}'.format(name))
            return relu, pooling
        else:
            return relu


def up_conv_layer(bottom, num_filters, kernel_size, strides, params, batch_norm=False, training=True,
                  out_size=None, name='', pad='valid'):
    with tf.compat.v1.variable_scope('UP_Conv_Layer_{}'.format(name)):
        up_conv = tf.compat.v1.layers.conv2d_transpose(
            inputs=bottom,
            filters=num_filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=None,
            padding=pad,
            kernel_regularizer=tf.keras.regularizers.L2(params['l2_reg_rate']),
            kernel_initializer=tf.keras.initializers.GlorotUniform(),  # tf.initializers.variance_scaling(scale=params['var_scale_factor'], distribution='uniform'),
            name='upconv{}'.format(name)
        )

        if batch_norm:
            up_conv = tf.compat.v1.layers.batch_normalization(up_conv, training=training, name='batch_norm_{}'.format(name))
            up_conv = tf.nn.relu(up_conv, name='relu_{}'.format(name))

        if out_size is not None:
            up_conv = crop_features(up_conv, out_size, name=name)

        return up_conv


def up_conv_add_layer(bottom, concat, params, kernel_size=4, num_filters=2, strides=2,
                      pad='valid', training=True, name=''):
    upconv = up_conv_layer(bottom, num_filters, kernel_size, strides, params, training=training,
                           name=name, pad=pad)

    with tf.compat.v1.variable_scope('Score_concat{}'.format(name)):
        out_size = concat.shape[1]
        upconv_shape = upconv.shape
        if upconv_shape[1] != out_size:
            upconv = crop_features(upconv, out_size, name=name)

        score_pool = tf.compat.v1.layers.conv2d(inputs=concat,
                                      filters=num_filters,
                                      kernel_size=1,
                                      padding=pad,
                                      data_format='channels_last',
                                      activation=None,
                                      kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                      name='score_layer')

        upconv = tf.add(upconv, score_pool)

        return upconv


def crop_features(features, out_size, name=''):
    with tf.compat.v1.name_scope('crop_{}'.format(name)):
        feat_shape = features.shape
        tf.cast(out_size, tf.int32)
        offsets = [0, tf.cast((int(feat_shape[1]) - int(out_size)) / 2, tf.int32),
                   tf.cast((int(feat_shape[2]) - int(out_size)) / 2, tf.int32), 0]
        size = [-1, out_size, out_size, feat_shape[3]]
        features = tf.slice(features, offsets, size, name='crop')
        return features


def upconv_concat_layer(bottom, concat, params, kernel_size=4, num_filters=2, strides=2,
                        pad='valid', training=True, name=''):
    upconv = up_conv_layer(bottom, num_filters, kernel_size, strides, params, batch_norm=True,
                           training=training, name=name, pad=pad)
    cropped = crop_features(concat, upconv.shape[1], name=name)
    return tf.concat([upconv, cropped], axis=-1, name='concat_{}'.format(name))


def resnet_base_layer(bottom, num_filters, strides):
    return False  # TODO: FINISH THIS IMPLEMENTATION
