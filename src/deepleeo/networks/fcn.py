import sys
from os import path
import tensorflow as tf
from importlib import reload

sys.path.insert(0, path.join(path.dirname(__file__),"../"))
import networks.layers as layers
import networks.loss_functions as lossf
reload(layers)
reload(lossf)


def fcn32_VGG_description(features, labels, params, mode, config):
    tf.logging.set_verbosity(tf.logging.INFO)
    training = mode == tf.estimator.ModeKeys.TRAIN

    hyper_params = params

    class_names = hyper_params["class_names"]
    num_channels = hyper_params["bands"]
    samples = features["data"]

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

    score_layer = tf.layers.conv2d(inputs=fconv7, filters=21, kernel_size=1, padding="same",
                                data_format="channels_last", activation=None, name="score_layer")

    #pred = tf.argmax(inputs=score_layer, dimension=3, name="pred")

    #TODO: Replace the '128' by the image size (height and width)
    up_score = layers.up_conv_layer(score_layer, filters=21, kernel_size=128, strides=32, name="uc")

    probs = tf.nn.softmax(up_score, axis=-1, name="pred_up")

    output = tf.argmax(probs, dimension=3, name="pred")

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=up_score)

    output = tf.cast(output, tf.float32)
    labels = tf.cast(labels, tf.float32)
    # loss = lossf.softmax_loss_cross_entropy(net_score=up_score, labels=labels, num_classes=len(class_names))
    loss = tf.losses.softmax_cross_entropy(tf.squeeze(labels), output)

    optimizer = tf.train.AdamOptimizer(learning_rate=hyper_params["learning_rate"], name="optimizer")

    # TODO: Review this piece of code
    # with tf.name_scope("metrics"):
    #     input_data_viz = (samples[:,:,:,0:num_channels] + 20)
    #     input_data_viz = tf.image.convert_image_dtype(input_data_viz, tf.uint8)
    #
    #     output_viz = tf.image.grayscale_to_rgb(output)
    #     labels_viz = tf.image.grayscale_to_rgb(labels)
    #
    #     tf.summary.image("img", input_data_viz, max_outputs=2)
    #     tf.summary.image("output", output_viz, max_outputs=2)
    #     tf.summary.image("labels", labels_viz, max_outputs=2)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    eval_summary_hook = tf.train.SummarySaverHook(save_steps=1,
                                                  output_dir=config.model_dir + "eval",
                                                  summary_op=tf.summary.merge_all())

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=up_score,
                                      loss=loss,
                                      train_op=train_op,
                                      evaluation_hooks=[eval_summary_hook])

def fcn_train(train_imgs, test_imgs, train_labels, test_labels, hyper_params, output_dir):
    tf.logging.set_verbosity(tf.logging.INFO)

    data_size, _, _, bands = train_imgs.shape
    hyper_params["bands"] = bands

    estimator = tf.estimator.Estimator(model_fn=fcn32_VGG_description,
                                       model_dir=output_dir,
                                       params=hyper_params)
    logging_hook = tf.train.LoggingTensorHook(tensors={}, every_n_iter=data_size)

    # for epoch in range(hyper_params["epochs"]):
    train_input = tf.estimator.inputs.numpy_input_fn(x={"data": train_imgs},
                                                     y=train_labels,
                                                     batch_size=hyper_params['batch_size'],
                                                     num_epochs=None,
                                                     shuffle=True)
    # images = tf.placeholder("float")
    # train_input = {images: train_imgs}
    # batch_images = tf.expand_dims(images, 0)

    train_results = estimator.train(input_fn=train_input, steps=hyper_params["epochs"], hooks=[logging_hook])

    test_input = tf.estimator.inputs.numpy_input_fn(x={"data": test_imgs},
                                                    y=test_labels,
                                                    num_epochs=None,
                                                    shuffle=False)

    test_results = estimator.evaluate(input_fn=test_input)