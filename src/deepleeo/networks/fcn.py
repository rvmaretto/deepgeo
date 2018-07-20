import tensorflow as tf

sys.path.insert(0, path.join(path.dirname(__file__),"../"))
import networks.layers as layers

def fcn_32(samples, labels, class_names, mode):
    g = tf.Graph()
    g.as_default()

    training = mode == tf.estimator.ModeKeys.TRAIN

    # Base Network (VGG_16)
    conv1_1 = layers.conv_pool_layer(input=samples, filters=64, training=training, name="1_1", pool=False)
    pool1 = layers.conv_pool_layer(inputs=conv1_1, filters=64, training=training, name="1_2")

    conv2_1 = layers.conv_pool_layer(inputs=pool1, filters=128, training=training, name="2_1", pool=False)
    pool2 = layers.conv_pool_layer(inputs=conv2_1, filters=128, training=training, name="2_2")

    conv3_1 = layers.conv_pool_layer(inputs=pool2, filters=256, training=training, name="3_1", pool=False)
    conv3_2 = layers.conv_pool_layer(inputs=conv3_1, filters=256, training=training, name="3_2", pool=False)
    pool3 = layers.conv_pool_layer(inputs=conv3_2, filters=256, training=training, name="3_3")

    conv4_1 = layers.conv_pool_layer(inputs=pool3, filters=512, training=training, name="4_1", pool=False)
    conv4_2 = layers.conv_pool_layer(inputs=conv4_1, filters=512, training=training, name="4_2", pool=False)
    pool4 = layers.conv_pool_layer(inputs=conv4_2, filters=512, training=training, name="4_3")

    conv5_1 = layers.conv_pool_layer(inputs=pool4, filters=512, training=training, name="5_1", pool=False)
    conv5_2 = layers.conv_pool_layer(inputs=conv5_1, filters=512, training=training, name="5_2", pool=False)
    pool5 = layers.conv_pool_layer(inputs=conv5_2, filters=512, training=training, name="5_3")

    # Fully Convolutional part
    fconv6 = layers.conv_pool_layer(inputs=pool5, filters=4096, kernel_size=7, training=training, name="fc6")
    if(training):
        fconv6 = tf.layers.dropout(inputs=fconv6, rate=0.5, name="drop_7")

    fconv7 = layers.conv_pool_layer(inputs=fconv6, filters=4096, kernel_size=1, training=training, name="fc7")
    if(training):
        fconv7 = tf.layers.dropout(inputs=fconv7, rate=0.5, name="drop_7")

    score_layer = tf.layers.conv2d(inputs=fconv7, filters=21, kernel_size=1, padding="same",
                                data_format="channels_last", activation=None, name="score_layer")

    pred = tf.argmax(inputs=score_layer, dimension=3, name="pred")

    up_score = layers.up_conv_layer(score_layer, filters=21, kernel_size=64, strides=32, name="uc")

    pred_up = tf.argmax(inputs=up_score, dimension=3, name="pred_up")

