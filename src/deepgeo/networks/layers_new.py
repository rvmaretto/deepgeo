import tensorflow as tf
import tensorflow.keras as keras

# TODO: Refactor. Re-implement this method in the following structure:
# conv_pool_layer(bottom, filters=[F1,F2,..., FN], poolings=[1,2,...,n], kernel_sizes=[k1, k2, ..., kn])
def conv_pool_layer(model, filters, kernel_size, params, pool=True, pad='same'):
    model.add(keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding=pad,
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        kernel_regularizer=tf.keras.regularizers.L2(params['l2_reg_rate']),
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
    ))
    model.add(keras.layers.BatchNorm2d(axis=3))
    model.add(keras.layers.Activation("relu"))
    if pool:
        model.add(keras.layers.MaxPool2d(
            pool_size=(2, 2),
            strides=2,
            padding='valid'
        ))

def up_conv_layer(model, filters, kernel_size, strides, params, batch_norm=False, out_size=None,
                  pad='valid'):
    model.add(keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=pad,
        kernel_regularizer=tf.keras.regularizers.L2(params['l2_reg_rate']),
        kernel_initializer=tf.keras.initializers.GlorotUniform()
    ))

    if batch_norm:
        model.add(keras.layers.BatchNorm2d(axis=3))
        model.add(keras.layers.Activation("relu"))

        ## CHECK IT
        # if out_size is not None:
        #     model.add(keras.layers.Cropping2D)
        #     up_conv = crop_features(up_conv, out_size, name=name)

def upconv_concat_layer(model, concat, params, kernel_size=4, num_filters=2, strides=2,
                        pad='valid', training=True, name=''):
    up_conv_layer(bottom, num_filters, kernel_size, strides, params, batch_norm=True,
                           training=training, name=name, pad=pad)
    cropped = crop_features(concat, upconv.shape[1], name=name)
    return tf.concat([upconv, cropped], axis=-1, name='concat_{}'.format(name))