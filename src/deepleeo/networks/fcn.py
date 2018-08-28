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


def fcn32_VGG_description(features, labels, params, mode, config):
    tf.logging.set_verbosity(tf.logging.INFO)
    training = mode == tf.estimator.ModeKeys.TRAIN
    evaluating = mode == tf.estimator.ModeKeys.EVAL

    hyper_params = params

    num_classes = len(hyper_params["class_names"])
    #num_channels = hyper_params["bands"]
    samples = features["data"]
    learning_rate = hyper_params["learning_rate"]

    height, width, _ = samples[0].shape

    if(training or evaluating):
        labels = tf.one_hot(tf.cast(tf.add(labels, -1), tf.uint8), num_classes)

    # Base Network (VGG_16)
    conv1_1 = layers.conv_pool_layer(inputs=samples, filters=64, training=training, name="1_1", pool=False)
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

    score_layer = tf.layers.conv2d(inputs=fconv7, filters=num_classes, kernel_size=1, padding="same",
                                data_format="channels_last", activation=None, name="score_layer")

    up_score = layers.up_conv_layer(score_layer, filters=num_classes, kernel_size=(height, width), strides=32, name="uc")

    probs = tf.nn.softmax(up_score, axis=-1, name="softmax")

    #output = tf.argmax(probs, axis=-1, name="argmax_prediction")
    #logits = tf.layers.conv2d(up_score, 1, (1, 1), name="output", activation=tf.nn.relu, padding="same",
    #                          kernel_initializer=tf.initializers.variance_scaling(scale=0.001, distribution="normal"))

    predictions = {
        "classes": tf.argmax(input=probs, axis=-1, name="Argmax_Prediction"),
        "probabilities": probs
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.softmax_cross_entropy(tf.squeeze(labels), probs)
    #loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.squeeze(labels), logits=output)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="Optimizer")
    optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

    # TODO: Review this piece of code
    # with tf.name_scope("metrics"):
    #     input_data_viz = (samples[:,:,:,0:3] + 20)
    #     input_data_viz = tf.image.convert_image_dtype(input_data_viz, tf.uint8)
    #
    #     output_viz = tf.image.grayscale_to_rgb(output)
    #     labels_viz = tf.image.grayscale_to_rgb(labels)
    #
    #     tf.summary.image("img", input_data_viz, max_outputs=2)
    #     tf.summary.image("output", output_viz, max_outputs=2)
    #     tf.summary.image("labels", labels_viz, max_outputs=2)

    # with tf.name_scope("accuracy"):
    #     accuracy = tf.metrics.accuracy(labels=labels, predictions=output)
    #     tf.summary.scalar("accuracy", accuracy)


    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    eval_summary_hook = tf.train.SummarySaverHook(save_steps=1,
                                                  output_dir=config.model_dir,
                                                  summary_op=tf.summary.merge_all())

    eval_metric_ops = {"accuracy": tf.metrics.accuracy(
        labels=tf.argmax(labels, axis=-1, name="Decode_one_hot"),
        predictions=predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions["classes"],
                                      loss=loss,
                                      train_op=train_op,
                                      evaluation_hooks=[eval_summary_hook],
                                      eval_metric_ops=eval_metric_ops)


def fcn_train(train_imgs, test_imgs, train_labels, test_labels, hyper_params, output_dir):
    tf.logging.set_verbosity(tf.logging.INFO)

    data_size, _, _, bands = train_imgs.shape
    hyper_params["bands"] = bands

    estimator = tf.estimator.Estimator(#model_fn=fcn32_VGG_description,
                                       model_fn=tf.contrib.estimator.replicate_model_fn(fcn32_VGG_description),
                                       model_dir=output_dir,
                                       params=hyper_params)
    logging_hook = tf.train.LoggingTensorHook(tensors={}, every_n_iter=25)

    for epoch in range(1, hyper_params["epochs"]+1):
        print("===============================================")
        print("Epoch ", epoch)
        train_input = tf.estimator.inputs.numpy_input_fn(x={"data": train_imgs},
                                                         y=train_labels,
                                                         batch_size=hyper_params["batch_size"],
                                                         num_epochs=1,
                                                         shuffle=True)

        print("---------------")
        print("Training...")
        train_results = estimator.train(input_fn=train_input, steps=None, hooks=[logging_hook])

        test_input = tf.estimator.inputs.numpy_input_fn(x={"data": test_imgs},
                                                        y=test_labels,
                                                        batch_size=hyper_params["batch_size"],
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

    estimator = tf.estimator.Estimator(model_fn=tf.contrib.estimator.replicate_model_fn(fcn32_VGG_description),
                                       model_dir=model_dir,
                                       params=params
    )

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
        #print(predict["probabilities"].shape)
        predicted_images.append(predict["classes"])

    return predicted_images