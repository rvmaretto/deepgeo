import sys
from os import path
import tensorflow as tf
import numpy as np
from importlib import reload

sys.path.insert(0, path.join(path.dirname(__file__),"../"))
import networks.layers as layers
import networks.loss_functions as lossf
import networks.tb_metrics as tbm
reload(layers)
reload(lossf)


def fcn32s_description(features, labels, params, mode, config):
    training = mode == tf.estimator.ModeKeys.TRAIN
    evaluating = mode == tf.estimator.ModeKeys.EVAL

    hyper_params = params

    num_classes = len(hyper_params["class_names"])
    #num_channels = hyper_params["bands"]
    samples = features["data"]
    learning_rate = hyper_params["learning_rate"]

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
        fconv6 = tf.layers.dropout(inputs=fconv6, rate=0.5, name="drop_6") # TODO: Put this rate in params

    # print("SHAPE FConv_6: ", fconv6.shape)

    fconv7 = layers.conv_pool_layer(bottom=fconv6, filters=4096, kernel_size=1, params=params, training=training,
                                    pool=False, name="fc7")
    if(training):
        fconv7 = tf.layers.dropout(inputs=fconv7, rate=0.5, name="drop_7") # TODO: Put this rate in params

    # print("SHAPE FConv_7: ", fconv7.shape)

    #TODO: Is it suitable to put the sigmoid here?
    score_layer = tf.layers.conv2d(inputs=fconv7, filters=num_classes, kernel_size=1, padding="same",
                                   data_format="channels_last", activation=None, name="score_layer")

    # print("SHAPE Score Layer: ", score_layer.shape)

    up_score = layers.up_conv_layer(score_layer, num_filters=num_classes, kernel_size=64, strides=32,
                                    params=params, out_size=height, pad="same", name="uc")

    # print("SHAPE Up Score: ", up_score.shape)

    # probs = tf.nn.softmax(up_score, axis=-1, name="softmax")
    #probs = tf.nn.sigmoid(up_score, name="sigmoid")

    # output = tf.argmax(probs, axis=-1, name="argmax_prediction")
    output = tf.layers.conv2d(up_score, 1, (1, 1), name="output", activation=tf.nn.sigmoid, padding="same",
                             kernel_initializer=tf.initializers.variance_scaling(scale=0.001, distribution="normal"))

    predictions = {
        "classes": output,#tf.argmax(input=up_score, axis=-1, name="Argmax_Prediction"),
        # "probabilities": probs
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # print("LABELS SHAPE: ", labels.shape)
    # print("OUTPUT SHAPE: ", output.shape)
    # labels_1hot = tf.one_hot(tf.cast(labels, tf.uint8), num_classes)
    # labels_1hot = tf.squeeze(labels_1hot)
    # loss = tf.losses.sigmoid_cross_entropy(labels_1hot, output)
    # loss = tf.losses.softmax_cross_entropy(labels_1hot, probs)
    # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.squeeze(labels), logits=output)
    # print(labels)
    # print(predictions["classes"])
    labels = tf.cast(labels, tf.float32)
    loss = lossf.twoclass_cost(predictions["classes"], labels)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="Optimizer")
    optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

    # labels2plot = tf.argmax(labels_1hot, axis=-1)

    tbm.plot_chips_tensorboard(samples, labels, output, bands_plot=params["bands_plot"],
                               num_chips=params['chips_tensorboard'])

    metrics, summaries = tbm.define_quality_metrics(labels, output, loss)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    # train_summary_hook = tf.train.SummarySaverHook(save_steps=1,
    #                                               output_dir=config.model_dir,
    #                                               summary_op=tf.summary.merge_all())

    eval_summary_hook = tf.train.SummarySaverHook(save_steps=10,
                                                  output_dir=config.model_dir + "/eval",
                                                  summary_op=tf.summary.merge_all())

    eval_metric_ops = {"eval_metrics/accuracy": metrics["accuracy"],
                       "eval_metrics/f1-score": metrics["f1_score"],
                       "eval_metrics/cross_entropy": metrics["cross_entropy"]}

    logging_hook = tf.train.LoggingTensorHook({#"batch_probs": probs,
                                               "batch_labels": labels,
                                               "batch_predictions": predictions["classes"]},
                                               # "unique_labels": unique_labels,
                                               # "unique_predictions": unique_predictions},
                                               every_n_iter=25)

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions["classes"],
                                      loss=loss,
                                      train_op=train_op,
                                      evaluation_hooks=[eval_summary_hook],
                                      eval_metric_ops=eval_metric_ops)
                                      # training_hooks=[logging_hook])