import sys
from os import path
import tensorflow as tf

sys.path.insert(0, path.join(path.dirname(__file__), ".."))
import networks.layers as layers

def unet_encoder(samples, num_classes, params, mode):
    training = mode == tf.estimator.ModeKeys.TRAIN


    #TODO: review the whole implementation, the number of filters and all the parameters
    with tf.variable_scope("Layer_1"):
        conv1_1 = layers.conv_pool_layer(bottom=samples, filters=64, params=params, training=training,
                                         name="1_1", pool=False)
        pool1 = layers.conv_pool_layer(bottom=conv1_1, filters=64, params=params, training=training,
                                       name="1_2")

    # print("SHAPE Conv_1: ", pool1.shape)

    with tf.variable_scope("Layer_2"):
        conv2_1 = layers.conv_pool_layer(bottom=pool1, filters=128, params=params, training=training,
                                         name="2_1", pool=False)
        pool2 = layers.conv_pool_layer(bottom=conv2_1, filters=128, params=params, training=training,
                                       name="2_2")

    # print("SHAPE Conv_2: ", pool2.shape)

    with tf.variable_scope("Layer_3"):
        conv3_1 = layers.conv_pool_layer(bottom=pool2, filters=256, params=params, training=training,
                                         name="3_1", pool=False)
        conv3_2 = layers.conv_pool_layer(bottom=conv3_1, filters=256, params=params, training=training,
                                         name="3_2", pool=False)
        pool3 = layers.conv_pool_layer(bottom=conv3_2, filters=256, params=params, training=training,
                                       name="3_3")

    # print("SHAPE Conv_3: ", pool3.shape)

    with tf.variable_scope("Layer_4"):
        conv4_1 = layers.conv_pool_layer(bottom=pool3, filters=512, params=params, training=training,
                                         name="4_1", pool=False)
        conv4_2 = layers.conv_pool_layer(bottom=conv4_1, filters=512, params=params, training=training,
                                         name="4_2", pool=False)
        pool4 = layers.conv_pool_layer(bottom=conv4_2, filters=512, params=params, training=training,
                                       name="4_3")

    # print("SHAPE Conv_4: ", pool4.shape)

    with tf.variable_scope("Layer_5"):
        conv5_1 = layers.conv_pool_layer(bottom=pool4, filters=512, params=params, training=training,
                                         name="5_1", pool=False)
        conv5_2 = layers.conv_pool_layer(bottom=conv5_1, filters=512, params=params, training=training,
                                         name="5_2", pool=False)
        pool5 = layers.conv_pool_layer(bottom=conv5_2, filters=512, params=params, training=training,
                                       name="5_3")

    # print("SHAPE Conv_5: ", pool5.shape)

    # Fully Convolutional part
    with tf.variable_scope("FC_Layer_1"):
        fconv6 = layers.conv_pool_layer(bottom=pool5, filters=4096, kernel_size=7, params=params,
                                        training=training, name="fc6", pool=False)
        if (training):
            fconv6 = tf.layers.dropout(inputs=fconv6, rate=params['dropout_rate'], name="drop_6")

    # print("SHAPE FConv_6: ", fconv6.shape)
    with tf.variable_scope("FC_Layer_2"):
        fconv7 = layers.conv_pool_layer(bottom=fconv6, filters=4096, kernel_size=1, params=params,
                                        training=training, name="fc7", pool=False)
        if (training):
            fconv7 = tf.layers.dropout(inputs=fconv7, rate=params['dropout_rate'], name="drop_7")

    # print("SHAPE FConv_7: ", fconv7.shape)

    # fconv8 = tf.layers.conv2d(inputs=fconv7, filters=1000, kernel_size=1, padding="same",
    #                             data_format="channels_last", activation=None, name="fc8")
    # if (training):
    #     fconv8 = tf.layers.dropout(inputs=fconv8, rate=0.5, name="drop_6")
    score_layer = tf.layers.conv2d(inputs=fconv7, filters=num_classes, kernel_size=1, padding="valid",
                                   data_format="channels_last", activation=None, name="Score_Layer_FC_2")


    return pool1, pool2, pool3, pool4, pool5, fconv6, fconv7, score_layer