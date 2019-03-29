import sys
from os import path
import tensorflow as tf

sys.path.insert(0, path.join(path.dirname(__file__), '..'))
import networks.layers as layers
# import networks.loss_functions as lossf
# import networks.tb_metrics as tbm

def unet_encoder(samples, params, mode, name_sufix=''):
    training = mode == tf.estimator.ModeKeys.TRAIN

    if params['fusion'] == 'early':
        total_channels = samples.get_shape().as_list()[3]
        num_channels = round(total_channels / 2)
        samples = tf.layers.conv2d(samples, filters=num_channels, kernel_size=(1,1), strides=1,
                                   padding='valid', activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                   name='time_fusion')

    # TODO: review the whole implementation, the number of filters and all the parameters
    conv_1 = layers.conv_pool_layer(bottom=samples, filters=64, params=params, training=training,
                                    name='1_1' + name_sufix, pool=False, pad='valid')
    conv_1_2, pool1 = layers.conv_pool_layer(bottom=conv_1, filters=64, params=params, training=training,
                                   name='1_2' + name_sufix, pad='valid')

    # print('SHAPE Conv_1: ', conv_1.shape)
    # print('SHAPE Pool_1: ', pool1.shape)

    conv_2 = layers.conv_pool_layer(bottom=pool1, filters=128, params=params, training=training,
                                    name='2_1' + name_sufix, pool=False, pad='valid')
    conv_2_1, pool2 = layers.conv_pool_layer(bottom=conv_2, filters=128, params=params, training=training,
                                   name='2_2' + name_sufix, pad='valid')

    # print('SHAPE Conv_2: ', conv_2.shape)
    # print('SHAPE Pool_2: ', pool2.shape)

    conv_3 = layers.conv_pool_layer(bottom=pool2, filters=256, params=params, training=training,
                                    name='3_1' + name_sufix, pool=False, pad='valid')
    conv_3_1, pool3 = layers.conv_pool_layer(bottom=conv_3, filters=256, params=params, training=training,
                                   name='3_2' + name_sufix, pad='valid')

    # print('SHAPE Conv_3: ', conv_3.shape)
    # print('SHAPE Pool_3: ', pool3.shape)

    conv_4 = layers.conv_pool_layer(bottom=pool3, filters=512, params=params, training=training,
                                    name='4_1' + name_sufix, pool=False, pad='valid')
    conv_4_1, pool4 = layers.conv_pool_layer(bottom=conv_4, filters=512, params=params, training=training,
                                   name='4_2' + name_sufix, pad='valid')

    # print('SHAPE Conv_4: ', conv_4.shape)
    # print('SHAPE Pool_4: ', pool4.shape)

    conv_5_1 = layers.conv_pool_layer(bottom=pool4, filters=1024, params=params, training=training,
                                      name='5_1' + name_sufix, pool=False, pad='valid')
    conv_5_2 = layers.conv_pool_layer(bottom=conv_5_1, filters=1024, params=params, training=training,
                                      name='5_2' + name_sufix, pool=False, pad='valid')

    # print('SHAPE Conv_5: ', conv_5_1.shape)
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

    dropout = tf.layers.dropout(conv_9_1, rate=params['dropout_rate'], training=training, name='dropout')

    logits = tf.layers.conv2d(dropout, params['num_classes'], (1, 1), activation=tf.nn.relu, padding='valid',
                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                              name='logits')

    # print('LOGITS SHAPE: ', logits.shape)

    return logits

def unet_description(samples, labels, params, mode, config):
    # tf.logging.set_verbosity(tf.logging.INFO)

    # learning_rate = params['learning_rate']
    # samples = features['data']

    height, width, _ = samples[0].shape

    encoded_feat = unet_encoder(samples, params, mode)
    logits = unet_decoder(encoded_feat, params, mode)

    return logits
    # if mode == tf.estimator.ModeKeys.PREDICT:
    #     return tf.estimator.EstimatorSpec(mode=mode, predictions=output)

    # cropped_labels = tf.cast(layers.crop_features(labels, output.shape[1], name='labels'), tf.float32)

    # cropped_labels = tf.cast(cropped_labels, tf.float32)
    # loss = lossf.twoclass_cost(output, cropped_labels)
    #
    # optimizer = tf.contrib.opt.NadamOptimizer(learning_rate, name='Optimizer')
    # optimizer = tf.contrib.estimator.TowerOptimizer(optimizer) #TODO: Verify if removing this it plots losses together
    #
    # tbm.plot_chips_tensorboard(samples, cropped_labels, output, bands_plot=params['bands_plot'],
    #                            num_chips=params['chips_tensorboard'])
    #
    # metrics, summaries = tbm.define_quality_metrics(cropped_labels, output, loss)
    #
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #
    # with tf.control_dependencies(update_ops):
    #     train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    #
    # # train_summary_hook = tf.train.SummarySaverHook(save_steps=1,
    # #                                               output_dir=config.model_dir,
    # #                                               summary_op=tf.summary.merge_all())
    #
    # eval_summary_hook = tf.train.SummarySaverHook(save_steps=10,
    #                                               output_dir=config.model_dir + '/eval',
    #                                               summary_op=tf.summary.merge_all())
    #
    # eval_metric_ops = {'eval_metrics/accuracy': metrics['accuracy'],
    #                    'eval_metrics/f1-score': metrics['f1_score'],
    #                    'eval_metrics/cross_entropy': metrics['cross_entropy']}
    #
    # return tf.estimator.EstimatorSpec(mode=mode,
    #                                   predictions=output,
    #                                   loss=loss,
    #                                   train_op=train_op,
    #                                   eval_metric_ops=eval_metric_ops,
    #                                   evaluation_hooks=[eval_summary_hook])
    #                                   # training_hooks=[train_summary_hook])
