import sys
from os import path
import tensorflow as tf
import numpy as np
from importlib import reload

sys.path.insert(0, path.join(path.dirname(__file__), '../'))
import networks.layers as layers
# import networks.loss_functions as lossf
# import networks.tb_metrics as tbm
reload(layers)
# reload(lossf)

# TODO: Refactor this file. Create a class and put the FCN8s, FCN16s and FCN32s in the same file/(function or class)
# TODO: Refactor this to allow multiple classes and to allow the user to chose the loss function through parameters.
def fcn1s_description(samples, labels, params, mode, config):
    training = mode == tf.estimator.ModeKeys.TRAIN
    evaluating = mode == tf.estimator.ModeKeys.EVAL

    num_classes = params['num_classes']
    learning_rate = params['learning_rate']
    # tf.identity(learning_rate, 'learning_rate')
    # tf.summary.scalar('learning_rate', learning_rate)

    height, width, _ = samples[0].shape

    # print('SHAPE LABELS: ', labels.shape)
    # print('SHAPE Input: ', samples.shape)
    # labels_1hot = labels

    # Base Network (VGG_16)
    conv1_1 = layers.conv_pool_layer(bottom=samples, filters=64, params=params, training=training, name='1_1',
                                     pool=False)
    conv1_2, pool1 = layers.conv_pool_layer(bottom=conv1_1, filters=64, params=params, training=training, name='1_2')

    # print('SHAPE Conv_1: ', pool1.shape)

    conv2_1 = layers.conv_pool_layer(bottom=pool1, filters=128, params=params, training=training,
                                     name='2_1', pool=False)
    conv2_2, pool2 = layers.conv_pool_layer(bottom=conv2_1, filters=128, params=params, training=training, name='2_2')

    # print('SHAPE Conv_2: ', pool2.shape)

    conv3_1 = layers.conv_pool_layer(bottom=pool2, filters=256, params=params, training=training,
                                     name='3_1', pool=False)
    conv3_2 = layers.conv_pool_layer(bottom=conv3_1, filters=256, params=params, training=training,
                                     name='3_2', pool=False)
    conv3_3, pool3 = layers.conv_pool_layer(bottom=conv3_2, filters=256, params=params, training=training, name='3_3')

    # print('SHAPE Conv_3: ', pool3.shape)

    conv4_1 = layers.conv_pool_layer(bottom=pool3, filters=512, params=params, training=training,
                                     name='4_1', pool=False)
    conv4_2 = layers.conv_pool_layer(bottom=conv4_1, filters=512, params=params, training=training,
                                     name='4_2', pool=False)
    conv4_3, pool4 = layers.conv_pool_layer(bottom=conv4_2, filters=512, params=params, training=training, name='4_3')

    # print('SHAPE Conv_4: ', pool4.shape)

    conv5_1 = layers.conv_pool_layer(bottom=pool4, filters=512, params=params, training=training,
                                     name='5_1', pool=False)
    conv5_2 = layers.conv_pool_layer(bottom=conv5_1, filters=512, params=params, training=training,
                                     name='5_2', pool=False)
    conv5_3, pool5 = layers.conv_pool_layer(bottom=conv5_2, filters=512, params=params, training=training, name='5_3')

    # print('SHAPE Conv_5: ', pool5.shape)

    # Fully Convolutional part
    fconv6 = layers.conv_pool_layer(bottom=pool5, filters=4096, kernel_size=7, params=params,
                                    training=training, name='fc6', pool=False)
    if(training):
        fconv6 = tf.compat.v1.layers.dropout(inputs=fconv6, rate=params['dropout_rate'], name='drop_6') # TODO: Put this rate in params

    # print('SHAPE FConv_6: ', fconv6.shape)
    fconv7 = layers.conv_pool_layer(bottom=fconv6, filters=4096, kernel_size=1, params=params,
                                    training=training, name='fc7', pool=False)
    if(training):
        fconv7 = tf.compat.v1.layers.dropout(inputs=fconv7, rate=params['dropout_rate'], name='drop_7') # TODO: Put this rate in params

    # print('SHAPE FConv_7: ', fconv7.shape)

    # fconv8 = tf.layers.conv2d(inputs=fconv7, filters=1000, kernel_size=1, padding='same',
    #                             data_format='channels_last', activation=None, name='fc8')

    # TODO: Is it suitable to put the sigmoid here? If yes, it should be used in the score pool too
    score_layer = tf.compat.v1.layers.conv2d(inputs=fconv7, filters=num_classes, kernel_size=1, padding='valid',
                                   data_format='channels_last', activation=None, name='Score_Layer_FC_2')

    if (training):
        score_layer = tf.compat.v1.layers.dropout(inputs=score_layer, rate=params['dropout_rate'], name='drop_8')


    up_score_1 = layers.up_conv_add_layer(score_layer, conv5_3, params=params, kernel_size=4,
                                             num_filters=num_classes, strides=2, pad='same', name='1')

    # print('SHAPE Up Score: ', up_score_1.shape)

    up_score_2 = layers.up_conv_add_layer(up_score_1, conv4_3, params=params, kernel_size=4,
                                             num_filters=num_classes, strides=2, pad='same', name='2')

    up_score_3 = layers.up_conv_add_layer(up_score_2, conv3_3, params=params, kernel_size=4,
                                          num_filters=num_classes, strides=2, pad='same', name='3')

    up_score_4 = layers.up_conv_add_layer(up_score_3, conv2_2, params=params, kernel_size=4,
                                          num_filters=num_classes, strides=2, pad='same', name='4')

    logits = layers.up_conv_add_layer(up_score_4, conv1_2, params=params, kernel_size=4,
                                        num_filters=num_classes, strides=2, pad='same', name='final')

    # print('SHAPE Up Score Final: ', up_final.shape)

    # output = tf.layers.conv2d(up_final, 1, (1, 1), name='output', activation=tf.nn.sigmoid, padding='same',
    #                          kernel_initializer=tf.initializers.variance_scaling(scale=0.001, distribution='uniform'))

    return logits