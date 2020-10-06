import sys
from os import path
import tensorflow as tf

sys.path.insert(0, path.join(path.dirname(__file__), '..'))
import networks.layers as layers
# import networks.loss_functions as lossf
# import networks.tb_metrics as tbm


def unet_encoder(samples, params, mode, name_sufix=''):
    training = mode == tf.estimator.ModeKeys.TRAIN

    # TODO: Remove this from here. Put it in the description, before calling the encoder.
    if 'fusion' in params:
        if params['fusion'] == 'early':
            total_channels = samples.get_shape().as_list()[3]
            num_channels = round(total_channels / 2)
            #total_channels = tf.shape(samples)[3]
            #num_channels = tf.cast(tf.round(total_channels / 2), tf.int32)
            samples = tf.compat.v1.layers.conv2d(samples, filters=num_channels, kernel_size=(1, 1), strides=1,
                                       padding='valid', activation=tf.nn.relu,
                                       kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                       name='time_fusion')
    # print('SHAPE INPUT: ', samples.shape)

    # TODO: review the whole implementation, the number of filters and all the parameters
    conv_1 = layers.conv_pool_layer(bottom=samples, filters=64, params=params, training=training,
                                    name='1_1' + name_sufix, pool=False, pad='valid')
    conv_1_2, pool1 = layers.conv_pool_layer(bottom=conv_1, filters=64, params=params, training=training,
                                   name='1_2' + name_sufix, pad='valid')

    # print('SHAPE Conv_1: ', conv_1.shape)
    # print('SHAPE Conv_1_2: ', conv_1_2.shape)
    # print('SHAPE Pool_1: ', pool1.shape)

    conv_2 = layers.conv_pool_layer(bottom=pool1, filters=128, params=params, training=training,
                                    name='2_1' + name_sufix, pool=False, pad='valid')
    conv_2_1, pool2 = layers.conv_pool_layer(bottom=conv_2, filters=128, params=params, training=training,
                                   name='2_2' + name_sufix, pad='valid')

    # print('SHAPE Conv_2: ', conv_2.shape)  
    # print('SHAPE Conv_2_1: ', conv_2_1.shape)
    # print('SHAPE Pool_2: ', pool2.shape)

    conv_3 = layers.conv_pool_layer(bottom=pool2, filters=256, params=params, training=training,
                                    name='3_1' + name_sufix, pool=False, pad='valid')
    conv_3_1, pool3 = layers.conv_pool_layer(bottom=conv_3, filters=256, params=params, training=training,
                                   name='3_2' + name_sufix, pad='valid')

    # print('SHAPE Conv_3: ', conv_3.shape)
    # print('SHAPE Conv_3_1: ', conv_3_1.shape)
    # print('SHAPE Pool_3: ', pool3.shape)

    conv_4 = layers.conv_pool_layer(bottom=pool3, filters=512, params=params, training=training,
                                    name='4_1' + name_sufix, pool=False, pad='valid')
    conv_4_1, pool4 = layers.conv_pool_layer(bottom=conv_4, filters=512, params=params, training=training,
                                   name='4_2' + name_sufix, pad='valid')

    # print('SHAPE Conv_4: ', conv_4.shape)
    # print('SHAPE Conv_4_1: ', conv_4_1.shape)
    # print('SHAPE Pool_4: ', pool4.shape)

    conv_5_1 = layers.conv_pool_layer(bottom=pool4, filters=1024, params=params, training=training,
                                      name='5_1' + name_sufix, pool=False, pad='valid')
    conv_5_2 = layers.conv_pool_layer(bottom=conv_5_1, filters=1024, params=params, training=training,
                                      name='5_2' + name_sufix, pool=False, pad='valid')

    # print('SHAPE Conv_5: ', conv_5_1.shape)
    # print('SHAPE Conv_5_1: ', conv_5_1.shape)
    # print('SHAPE Pool_5: ', conv_5_2.shape)

    return {'conv_1': conv_1_2,
            'conv_2': conv_2_1,
            'conv_3': conv_3_1,
            'conv_4': conv_4_1,
            'conv_5': conv_5_2}


def unet_decoder(features, params, mode):
    training = mode == tf.estimator.ModeKeys.TRAIN

    up6 = layers.upconv_concat_layer(features['conv_5'], features['conv_4'], params, num_filters=512,
                                      kernel_size=2, strides=2, pad='valid', training=training, name='6')
    conv_6 = layers.conv_pool_layer(up6, filters=512, params=params, kernel_size=3, training=training,
                                    pool=False, pad='valid', name='6')
    conv_6_1 = layers.conv_pool_layer(conv_6, filters=512, params=params, kernel_size=3, training=training,
                                      pool=False, pad='valid', name='6_1')

    # print('SHAPE up6: ', up6.shape)
    # print('SHAPE conv_6: ', conv_6.shape)
    # print('SHAPE conv_6_1: ', conv_6_1.shape)

    up7 = layers.upconv_concat_layer(conv_6_1, features['conv_3'], params, num_filters=256,
                                      kernel_size=2, strides=2, pad='valid', training=training, name='7')
    conv_7 = layers.conv_pool_layer(up7, filters=256, params=params, kernel_size=3, training=training,
                                    pool=False, pad='valid', name='7')
    conv_7_1 = layers.conv_pool_layer(conv_7, filters=256, params=params, kernel_size=3, training=training,
                                      pool=False, pad='valid', name='7_1')

    # print('SHAPE up7: ', up7.shape)
    # print('SHAPE conv_7: ', conv_7.shape)
    # print('SHAPE conv_7_1: ', conv_7_1.shape)

    up8 = layers.upconv_concat_layer(conv_7_1, features['conv_2'], params, num_filters=128,
                                      kernel_size=2, strides=2, pad='valid', training=training, name='8')
    conv_8 = layers.conv_pool_layer(up8, filters=128, params=params, kernel_size=3, training=training,
                                    pool=False, pad='valid', name='8')
    conv_8_1 = layers.conv_pool_layer(conv_8, filters=128, params=params, kernel_size=3, training=training,
                                      pool=False, pad='valid', name='8_1')

    # print('SHAPE up8: ', up8.shape)
    # print('SHAPE conv_8: ', conv_8.shape)
    # print('SHAPE conv_8_1: ', conv_8_1.shape)

    up9 = layers.upconv_concat_layer(conv_8_1, features['conv_1'], params, num_filters=64,
                                      kernel_size=2, strides=2, pad='valid', training=training, name='9')
    conv_9 = layers.conv_pool_layer(up9, filters=64, params=params, kernel_size=3, training=training,
                                    pool=False, pad='valid', name='9')
    conv_9_1 = layers.conv_pool_layer(conv_9, filters=64, params=params, kernel_size=3, training=training,
                                      pool=False, pad='valid', name='9_1')

    # print('SHAPE up9: ', up9.shape)
    # print('SHAPE conv_9: ', conv_9.shape)
    # print('SHAPE conv_9_1: ', conv_9_1.shape)

    # dropout = tf.layers.dropout(conv_9_1, rate=params['dropout_rate'], training=training, name='dropout')

    # logits = tf.layers.conv2d(conv_9_1, params['num_classes'], (1, 1), activation=tf.nn.relu, padding='valid',
    #                           kernel_initializer=tf.keras.initializers.GlorotUniform(),
    #                           name='logits')

    # print('LOGITS SHAPE: ', logits.shape)

    return conv_9_1


def unet_description(samples, labels, params, mode, config):
    # tf.logging.set_verbosity(tf.logging.INFO)

    # learning_rate = params['learning_rate']
    # samples = features['data']

    height, width, _ = samples[0].shape

    encoded_feat = unet_encoder(samples, params, mode)
    last_conv = unet_decoder(encoded_feat, params, mode)

    logits = tf.compat.v1.layers.conv2d(last_conv, params['num_classes'], (1, 1), activation=tf.nn.relu, padding='valid',
                              kernel_initializer=tf.keras.initializers.GlorotUniform(),
                              name='logits')

    return logits
