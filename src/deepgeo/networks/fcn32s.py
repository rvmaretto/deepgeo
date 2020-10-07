import sys
from os import path
import tensorflow as tf
import numpy as np
from importlib import reload

sys.path.insert(0, path.join(path.dirname(__file__),"../"))
import networks.layers as layers
# import networks.loss_functions as lossf
# import networks.tb_metrics as tbm
reload(layers)
# reload(lossf)


def fcn32s_description(samples, labels, params, mode, config):
    training = mode == tf.estimator.ModeKeys.TRAIN
    evaluating = mode == tf.estimator.ModeKeys.EVAL

    num_classes = params['num_classes']

    height, width, _ = samples[0].shape

    # print("SHAPE LABELS: ", labels.shape)
    # print("SHAPE Input: ", samples.shape)
    # labels_1hot = labels

    # Base Network (VGG_16)
    conv1_1 = layers.conv_pool_layer(bottom=samples, filters=64, params=params, training=training, name="1_1",
                                     pool=False)
    conv1_2, pool1 = layers.conv_pool_layer(bottom=conv1_1, filters=64, params=params, training=training, name="1_2")

    # print("SHAPE Conv_1: ", pool1.shape)

    conv2_1 = layers.conv_pool_layer(bottom=pool1, filters=128, params=params, training=training, name="2_1",
                                     pool=False)
    conv2_1, pool2 = layers.conv_pool_layer(bottom=conv2_1, filters=128, params=params, training=training, name="2_2")

    # print("SHAPE Conv_2: ", pool2.shape)

    conv3_1 = layers.conv_pool_layer(bottom=pool2, filters=256, params=params, training=training, name="3_1",
                                     pool=False)
    conv3_2 = layers.conv_pool_layer(bottom=conv3_1, filters=256, params=params, training=training, name="3_2",
                                     pool=False)
    conv3_3, pool3 = layers.conv_pool_layer(bottom=conv3_2, filters=256, params=params, training=training, name="3_3")

    # print("SHAPE Conv_3: ", pool3.shape)

    conv4_1 = layers.conv_pool_layer(bottom=pool3, filters=512, params=params, training=training, name="4_1",
                                     pool=False)
    conv4_2 = layers.conv_pool_layer(bottom=conv4_1, filters=512, params=params, training=training, name="4_2",
                                     pool=False)
    conv4_3, pool4 = layers.conv_pool_layer(bottom=conv4_2, filters=512, params=params, training=training, name="4_3")

    # print("SHAPE Conv_4: ", pool4.shape)

    conv5_1 = layers.conv_pool_layer(bottom=pool4, filters=512, params=params, training=training, name="5_1",
                                     pool=False)
    conv5_2 = layers.conv_pool_layer(bottom=conv5_1, filters=512, params=params, training=training, name="5_2",
                                     pool=False)
    conv5_3, pool5 = layers.conv_pool_layer(bottom=conv5_2, filters=512, params=params, training=training, name="5_3")

    # print("SHAPE Conv_5: ", pool5.shape)

    # Fully Convolutional part
    fconv6 = layers.conv_pool_layer(bottom=pool5, filters=4096, kernel_size=7, params=params, training=training,
                                    pool=False, name="fc6")
    if(training):
        fconv6 = tf.compat.v1.layers.dropout(inputs=fconv6, rate=params['dropout_rate'], name="drop_6")

    # print("SHAPE FConv_6: ", fconv6.shape)

    fconv7 = layers.conv_pool_layer(bottom=fconv6, filters=4096, kernel_size=1, params=params, training=training,
                                    pool=False, name="fc7")
    if(training):
        fconv7 = tf.compat.v1.layers.dropout(inputs=fconv7, rate=params['dropout_rate'], name="drop_7")

    # print("SHAPE FConv_7: ", fconv7.shape)

    score_layer = tf.compat.v1.layers.conv2d(inputs=fconv7, filters=num_classes, kernel_size=1, padding="same",
                                   data_format="channels_last", activation=None, name="score_layer")

    # print("SHAPE Score Layer: ", score_layer.shape)

    logits = layers.up_conv_layer(score_layer, num_filters=num_classes, kernel_size=64, strides=32,
                                    params=params, out_size=height, pad="same", name="uc")

    return logits