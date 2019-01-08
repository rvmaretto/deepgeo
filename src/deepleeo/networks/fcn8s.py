import sys
from os import path
import tensorflow as tf
import numpy as np
from importlib import reload

sys.path.insert(0, path.join(path.dirname(__file__),"../"))
import networks.layers as layers
import networks.loss_functions as lossf
reload(layers)
reload(lossf)

#TODO: Refactor this file. Create a class and put the FCN8s, FCN16s and FCN32s in the same file/(function or class)
#TODO: Refactor this to allow multiple classes and to allow the user to chose the loss function through parameters.
def fcn8s_description(features, labels, params, mode, config):
    tf.logging.set_verbosity(tf.logging.INFO)
    training = mode == tf.estimator.ModeKeys.TRAIN
    evaluating = mode == tf.estimator.ModeKeys.EVAL

    num_classes = len(params["class_names"])
    #num_channels = hyper_params["bands"]
    samples = features["data"]
    learning_rate = params["learning_rate"]
    # tf.identity(learning_rate, "learning_rate")
    # tf.summary.scalar('learning_rate', learning_rate)

    height, width, _ = samples[0].shape

    # print("SHAPE LABELS: ", labels.shape)
    # print("SHAPE Input: ", samples.shape)
    # labels_1hot = labels

    # Base Network (VGG_16)
    with tf.variable_scope("Layer_1"):
        conv1_1 = layers.conv_pool_layer(bottom=samples, filters=64, params=params, training=training, name="1_1",
                                         pool=False)
        pool1 = layers.conv_pool_layer(bottom=conv1_1, filters=64, params=params, training=training, name="1_2")

    # print("SHAPE Conv_1: ", pool1.shape)

    with tf.variable_scope("Layer_2"):
        conv2_1 = layers.conv_pool_layer(bottom=pool1, filters=128, params=params, training=training,
                                         name="2_1", pool=False)
        pool2 = layers.conv_pool_layer(bottom=conv2_1, filters=128, params=params, training=training, name="2_2")

    # print("SHAPE Conv_2: ", pool2.shape)

    with tf.variable_scope("Layer_3"):
        conv3_1 = layers.conv_pool_layer(bottom=pool2, filters=256, params=params, training=training,
                                         name="3_1", pool=False)
        conv3_2 = layers.conv_pool_layer(bottom=conv3_1, filters=256, params=params, training=training,
                                         name="3_2", pool=False)
        pool3 = layers.conv_pool_layer(bottom=conv3_2, filters=256, params=params, training=training, name="3_3")

    # print("SHAPE Conv_3: ", pool3.shape)

    with tf.variable_scope("Layer_4"):
        conv4_1 = layers.conv_pool_layer(bottom=pool3, filters=512, params=params, training=training,
                                         name="4_1", pool=False)
        conv4_2 = layers.conv_pool_layer(bottom=conv4_1, filters=512, params=params, training=training,
                                         name="4_2", pool=False)
        pool4 = layers.conv_pool_layer(bottom=conv4_2, filters=512, params=params, training=training, name="4_3")

    # print("SHAPE Conv_4: ", pool4.shape)

    with tf.variable_scope("Layer_5"):
        conv5_1 = layers.conv_pool_layer(bottom=pool4, filters=512, params=params, training=training,
                                         name="5_1", pool=False)
        conv5_2 = layers.conv_pool_layer(bottom=conv5_1, filters=512, params=params, training=training,
                                         name="5_2", pool=False)
        pool5 = layers.conv_pool_layer(bottom=conv5_2, filters=512, params=params, training=training, name="5_3")

    # print("SHAPE Conv_5: ", pool5.shape)

    # Fully Convolutional part
    with tf.variable_scope("FC_Layer_1"):
        fconv6 = layers.conv_pool_layer(bottom=pool5, filters=4096, kernel_size=7, params=params,
                                        training=training, name="fc6", pool=False)
        if(training):
            fconv6 = tf.layers.dropout(inputs=fconv6, rate=params["dropout_rate"], name="drop_6") # TODO: Put this rate in params

    # print("SHAPE FConv_6: ", fconv6.shape)
    with tf.variable_scope("FC_Layer_2"):
        fconv7 = layers.conv_pool_layer(bottom=fconv6, filters=4096, kernel_size=1, params=params,
                                        training=training, name="fc7", pool=False)
        if(training):
            fconv7 = tf.layers.dropout(inputs=fconv7, rate=params["dropout_rate"], name="drop_7") # TODO: Put this rate in params

    # print("SHAPE FConv_7: ", fconv7.shape)

    # fconv8 = tf.layers.conv2d(inputs=fconv7, filters=1000, kernel_size=1, padding="same",
    #                             data_format="channels_last", activation=None, name="fc8")
    
    # TODO: Is it suitable to put the sigmoid here? If yes, it should be used in the score pool too
    score_layer = tf.layers.conv2d(inputs=fconv7, filters=num_classes, kernel_size=1, padding="valid",
                                   data_format="channels_last", activation=None, name="Score_Layer_FC_2")

    if (training):
        score_layer = tf.layers.dropout(inputs=score_layer, rate=params["dropout_rate"], name="drop_8")

    # score_pool4 = tf.layers.conv2d(inputs=pool5, filters=num_classes, kernel_size=1, padding="same",
    #                                data_format="channels_last", activation=None, name="score_pool4")

    # print("SHAPE Score Layer: ", score_layer.shape)

    # up_score_1 = layers.up_conv_layer(score_layer, filters=num_classes,
    #                                   kernel_size=(height - 64, width - 64),
    #                                   strides=32, name="uc")
    up_score_1 = layers.up_conv_add_layer(score_layer, pool4, params=params, kernel_size=4,
                                             num_filters=num_classes, strides=2, pad="same", name="1")

    # print("SHAPE Up Score: ", up_score_1.shape)

    up_score_2 = layers.up_conv_add_layer(up_score_1, pool3, params=params, kernel_size=4,
                                             num_filters=num_classes, strides=2, pad="same", name="2")

    up_score_3 = layers.up_conv_add_layer(up_score_2, pool2, params=params, kernel_size=4,
                                          num_filters=num_classes, strides=2, pad="same", name="3")

    up_final = layers.up_conv_layer(up_score_3, num_filters=num_classes, kernel_size=8, strides=4,
                                    params=params, out_size=height, pad="same", name="final")

    # print("SHAPE Up Score Final: ", up_final.shape)

    # probs = tf.nn.softmax(up_score_1, axis=-1, name="softmax")
    # probs = tf.nn.sigmoid(up_score_1, name="sigmoid")
    # output = tf.argmax(probs, axis=-1, name="argmax_prediction")

    output = tf.layers.conv2d(up_final, 1, (1, 1), name="output", activation=tf.nn.sigmoid, padding="same",
                             kernel_initializer=tf.initializers.variance_scaling(scale=0.001, distribution="uniform"))

    predictions = {
        "classes": output,#tf.argmax(input=up_score_1, axis=-1, name="Argmax_Prediction"),
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

    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="Optimizer")
    optimizer = tf.contrib.opt.NadamOptimizer(learning_rate, name="Optimizer")
    optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

    # labels2plot = tf.argmax(labels_1hot, axis=-1)

    with tf.name_scope("image_metrics"):
        input_data_vis = layers.crop_features(samples, output.shape[1])
        bands = tf.constant(params['bands_plot'])
        input_data_vis = tf.transpose(tf.nn.embedding_lookup(tf.transpose(input_data_vis), bands))
        input_data_vis = tf.image.convert_image_dtype(input_data_vis, tf.uint8, saturate=True)

        # labels_vis = tf.cast(labels, tf.float32)
        labels_vis = tf.image.grayscale_to_rgb(labels)

        output_vis = tf.cast(predictions["classes"], tf.float32)
        output_vis = tf.image.grayscale_to_rgb(output_vis)

        # labels2plot_vis = tf.image.convert_image_dtype(labels2plot, tf.uint8)
        # labels2plot_vis = tf.image.grayscale_to_rgb(tf.expand_dims(labels2plot_vis, axis=-1))

        tf.summary.image("input_image", input_data_vis, max_outputs=params['chips_tensorboard'])
        tf.summary.image("output", output_vis, max_outputs=params['chips_tensorboard'])
        tf.summary.image("labels", labels_vis, max_outputs=params['chips_tensorboard'])
        # tf.summary.image("labels1hot", labels2plot_vis, max_outputs=4)


    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    # train_summary_hook = tf.train.SummarySaverHook(save_steps=1,
    #                                               output_dir=config.model_dir,
    #                                               summary_op=tf.summary.merge_all())

    eval_summary_hook = tf.train.SummarySaverHook(save_steps=1,
                                                  output_dir=config.model_dir+"/eval", #TODO: When I change this, it start to plot also the eval chips
                                                  summary_op=tf.summary.merge_all())

    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    eval_metric_ops = {"accuracy": accuracy}
    # tf.identity(accuracy, "accuracy")
    # tf.summary.scalar("accuracy", accuracy[1])

    # logging_hook = tf.train.LoggingTensorHook({#"batch_probs": probs,
    #                                            "batch_labels": labels,
    #                                            "batch_predictions": predictions["classes"]},
    #                                            # "unique_labels": unique_labels,
    #                                            # "unique_predictions": unique_predictions},
    #                                            every_n_iter=25)

    #TODO: Review this: How to plot both the train and evaluation loss in the same graph?
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions["classes"],
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops,
                                      evaluation_hooks=[eval_summary_hook])#,
                                      # training_hooks=[train_summary_hook])