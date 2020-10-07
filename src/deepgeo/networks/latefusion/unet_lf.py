import sys
from os import path
import tensorflow as tf

sys.path.insert(0, path.join(path.dirname(__file__), ".."))
import networks.unet as unet
# import networks.layers as layers
# import networks.loss_functions as lossf
# import networks.tb_metrics as tbm

def unet_lf_description(samples, labels, params, mode, config):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    # learning_rate = params["learning_rate"]
    # samples = features["data"]
    # timesteps = samples.shape[4]
    timesteps = 2
    # samples = []
    # bands_1 = tf.constant([0,1,2,3,4])
    # bands_2 = tf.constant([5,6,7,8,9])
    # samples.append(tf.transpose(tf.nn.embedding_lookup(tf.transpose(features["data"]), bands_1)))
    # samples.append(tf.transpose(tf.nn.embedding_lookup(tf.transpose(features["data"]), bands_2)))
    # samples.append(features["data"][:,:,:,0:5])
    # samples.append(features["data"][:,:,:,5:10])
    samples_t1 = samples[:,:,:,0:5]
    samples_t2 = samples[:,:,:,5:10]

    height, width, _ = samples[0].shape

    # print(samples.shape)

    # out_encoder = []
    # for i in range(timesteps):
    #     name_sufix = "t_" + str(i)
    #     out_encoder.append(unet.unet_encoder(samples[i], params, mode, name_sufix))
    convs_t1 = unet.unet_encoder(samples_t1, params, mode, "t_1")
    convs_t2 = unet.unet_encoder(samples_t2, params, mode, "t_2")

    encoded_feat = {}
    # TODO: Encapsulate the following scopes in one for loop
    with tf.compat.v1.name_scope('Fusion_1'):
        encoded_feat['conv_1'] = tf.concat([convs_t1['conv_1'], convs_t2['conv_1']], axis=-1, name='concat_t1')
        encoded_feat['conv_1'] = tf.compat.v1.layers.conv2d(encoded_feat['conv_1'], filters=64, kernel_size=(1,1), strides=1,
                                                  padding='valid', activation=tf.nn.relu,
                                                  kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                                  name='conv_fusion_1')

    with tf.compat.v1.name_scope('Fusion_2'):
        encoded_feat['conv_2'] = tf.concat([convs_t1['conv_2'], convs_t2['conv_2']], axis=-1, name='concat_t2')
        encoded_feat['conv_2'] = tf.compat.v1.layers.conv2d(encoded_feat['conv_2'], filters=128, kernel_size=(1, 1), strides=1,
                                                  padding='valid', activation=tf.nn.relu,
                                                  kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                                  name='conv_fusion_2')

    with tf.compat.v1.name_scope('Fusion_3'):
        encoded_feat['conv_3'] = tf.concat([convs_t1['conv_3'], convs_t2['conv_3']], axis=-1, name='concat_t3')
        encoded_feat['conv_3'] = tf.compat.v1.layers.conv2d(encoded_feat['conv_3'], filters=256, kernel_size=(1, 1), strides=1,
                                                  padding='valid', activation=tf.nn.relu,
                                                  kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                                  name='conv_fusion_3')

    with tf.compat.v1.name_scope('Fusion_4'):
        encoded_feat['conv_4'] = tf.concat([convs_t1['conv_4'], convs_t2['conv_4']], axis=-1, name='concat_t4')
        encoded_feat['conv_4'] = tf.compat.v1.layers.conv2d(encoded_feat['conv_4'], filters=512, kernel_size=(1, 1), strides=1,
                                                  padding='valid', activation=tf.nn.relu,
                                                  kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                                  name='conv_fusion_4')

    with tf.compat.v1.name_scope('Fusion_5'):
        encoded_feat['conv_5'] = tf.concat([convs_t1['conv_5'], convs_t2['conv_5']], axis=-1, name='concat_t5')
        encoded_feat['conv_5'] = tf.compat.v1.layers.conv2d(encoded_feat['conv_5'], filters=1024, kernel_size=(1, 1), strides=1,
                                                  padding='valid', activation=tf.nn.relu,
                                                  kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                                  name='conv_fusion_5')


    # encoded_feat = out_encoder[0]
    # for i in range(1, len(out_encoder)):
    #     for k, feat in out_encoder[i].items():
    #         encoded_feat[k] = tf.concat([encoded_feat[k], feat], axis=-1, name="Fusion_{}".format(k))

    last_conv = unet.unet_decoder(encoded_feat, params, mode)

    logits = tf.compat.v1.layers.conv2d(last_conv, params['num_classes'], (1, 1), activation=tf.nn.relu, padding='valid',
                              kernel_initializer=tf.keras.initializers.GlorotUniform(),
                              name='logits')

    # print(logits.shape)

    return logits