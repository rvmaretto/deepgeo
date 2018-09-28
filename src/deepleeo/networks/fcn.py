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


def fcn8_description(features, labels, params, mode, config):
    tf.logging.set_verbosity(tf.logging.INFO)
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
    conv1_1 = layers.conv_pool_layer(inputs=samples, filters=64, training=training, name="1_1", pool=False)
    pool1 = layers.conv_pool_layer(inputs=conv1_1, filters=64, training=training, name="1_2")

    # print("SHAPE Conv_1: ", pool1.shape)

    conv2_1 = layers.conv_pool_layer(inputs=pool1, filters=128, training=training, name="2_1", pool=False)
    pool2 = layers.conv_pool_layer(inputs=conv2_1, filters=128, training=training, name="2_2")

    # print("SHAPE Conv_2: ", pool2.shape)

    conv3_1 = layers.conv_pool_layer(inputs=pool2, filters=256, training=training, name="3_1", pool=False)
    conv3_2 = layers.conv_pool_layer(inputs=conv3_1, filters=256, training=training, name="3_2", pool=False)
    pool3 = layers.conv_pool_layer(inputs=conv3_2, filters=256, training=training, name="3_3")

    # print("SHAPE Conv_3: ", pool3.shape)

    conv4_1 = layers.conv_pool_layer(inputs=pool3, filters=512, training=training, name="4_1", pool=False)
    conv4_2 = layers.conv_pool_layer(inputs=conv4_1, filters=512, training=training, name="4_2", pool=False)
    pool4 = layers.conv_pool_layer(inputs=conv4_2, filters=512, training=training, name="4_3")

    # print("SHAPE Conv_4: ", pool4.shape)

    conv5_1 = layers.conv_pool_layer(inputs=pool4, filters=512, training=training, name="5_1", pool=False)
    conv5_2 = layers.conv_pool_layer(inputs=conv5_1, filters=512, training=training, name="5_2", pool=False)
    pool5 = layers.conv_pool_layer(inputs=conv5_2, filters=512, training=training, name="5_3")

    # print("SHAPE Conv_5: ", pool5.shape)

    # Fully Convolutional part
    fconv6 = layers.conv_pool_layer(inputs=pool5, filters=4096, kernel_size=7, training=training, name="fc6")
    if(training):
        fconv6 = tf.layers.dropout(inputs=fconv6, rate=0.5, name="drop_6")

    # print("SHAPE FConv_6: ", fconv6.shape)

    fconv7 = layers.conv_pool_layer(inputs=fconv6, filters=4096, kernel_size=1, training=training, name="fc7")
    if(training):
        fconv7 = tf.layers.dropout(inputs=fconv7, rate=0.5, name="drop_7")

    # print("SHAPE FConv_7: ", fconv7.shape)

    score_layer = tf.layers.conv2d(inputs=fconv7, filters=1000, kernel_size=1, padding="same",
                                data_format="channels_last", activation=None, name="score_layer")

    # print("SHAPE Score Layer: ", score_layer.shape)

    up_score = layers.up_conv_layer(score_layer, filters=num_classes,
                                    kernel_size=(height - 64, width - 64), strides=32, name="uc")

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

    with tf.name_scope("metrics"):
        input_data_vis = (samples[:,:,:,1:4])
        input_data_vis = tf.image.convert_image_dtype(input_data_vis, tf.uint8, saturate=True)

        # labels_vis = tf.cast(labels, tf.float32)
        labels_vis = tf.image.grayscale_to_rgb(labels)

        output_vis = tf.cast(predictions["classes"], tf.float32)
        output_vis = tf.image.grayscale_to_rgb(output_vis)

        # labels2plot_vis = tf.image.convert_image_dtype(labels2plot, tf.uint8)
        # labels2plot_vis = tf.image.grayscale_to_rgb(tf.expand_dims(labels2plot_vis, axis=-1))

        tf.summary.image("input_image", input_data_vis, max_outputs=4)
        tf.summary.image("output", output_vis, max_outputs=4)
        tf.summary.image("labels", labels_vis, max_outputs=4)
        # tf.summary.image("labels1hot", labels2plot_vis, max_outputs=4)


    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    eval_summary_hook = tf.train.SummarySaverHook(save_steps=1,
                                                  output_dir=config.model_dir,
                                                  summary_op=tf.summary.merge_all())

    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels,
                                                       predictions=predictions["classes"])
    }

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


def fcn_train(train_imgs, test_imgs, train_labels, test_labels, params, output_dir):
    # tf.set_random_seed(1987)
    tf.logging.set_verbosity(tf.logging.INFO)

    data_size, _, _, bands = train_imgs.shape
    params["bands"] = bands

    # print("UNIQUE LABELS: ", np.unique(train_labels))
    # print("UNIQUE IMAGE: ", np.unique(train_imgs))

    estimator = tf.estimator.Estimator(#model_fn=fcn32_VGG_description,
                                       model_fn=tf.contrib.estimator.replicate_model_fn(fcn8_description),
                                       model_dir=output_dir,
                                       params=params)
    logging_hook = tf.train.LoggingTensorHook(tensors={}, every_n_iter=25)

    for epoch in range(1, params["epochs"] + 1):
        print("===============================================")
        print("Epoch ", epoch)
        train_input = tf.estimator.inputs.numpy_input_fn(x={"data": train_imgs},
                                                         y=train_labels,
                                                         batch_size=params["batch_size"],
                                                         num_epochs=1,
                                                         shuffle=True)

        print("---------------")
        print("Training...")
        train_results = estimator.train(input_fn=train_input, steps=None, hooks=[logging_hook])

        test_input = tf.estimator.inputs.numpy_input_fn(x={"data": test_imgs},
                                                        y=test_labels,
                                                        batch_size=params["batch_size"],
                                                        num_epochs=1,
                                                        shuffle=False)

        print("---------------")
        print("Evaluating...")
        test_results = estimator.evaluate(input_fn=test_input, name="Evaluation")

# def fcn_evaluate(images, labels, params, model_dir):
#     data_size, _, _, _ = images.shape
#
#     tf.logging.set_verbosity(tf.logging.WARN)
#     
#     estimator = tf.estimator.Estimator(model_fn=tf.contrib.estimator.replicate_model_fn(fcn32_VGG_description),
#                                        model_dir=model_dir,
#                                        params=params)
#     logging_hook = tf.train.LoggingTensorHook(tensors={}, every_n_iter=data_size)
#
#     input_imgs

def fcn_predict(images, params, model_dir):
    tf.logging.set_verbosity(tf.logging.WARN)

    if params["multi_gpu"]:
        estimator = tf.estimator.Estimator(model_fn=tf.contrib.estimator.replicate_model_fn(fcn8_description),
                                           model_dir=model_dir,
                                           params=params)
    else:
        estimator = tf.estimator.Estimator(model_fn=fcn8_description,
                                           model_dir=model_dir,
                                           params=params)

    if not isinstance(images, np.ndarray):
        images = np.stack(images).astype(np.float32)

    data_size, _, _ ,_ = images.shape
    input_fn = tf.estimator.inputs.numpy_input_fn(x={"data": images},
                                                  batch_size=params["batch_size"],
                                                  shuffle=False)

    predictions = estimator.predict(input_fn=input_fn)

    print("Classifying image with structure ", str(images.shape), "...")

    predicted_images = []

    for predict, dummy in zip(predictions, images):
        # predicted_images.append(np.argmax(predict["probabilities"], -1))
        # classif = np.argmax(predict["probabilities"], axis=-1)
        predicted_images.append(predict["classes"])


    return predicted_images