import sys
from os import path
import tensorflow as tf

sys.path.insert(0, path.join(path.dirname(__file__), ".."))
import networks.unet as unet
import networks.layers as layers
import networks.loss_functions as lossf

def unet_lf_description(features, labels, params, mode, config):
    tf.logging.set_verbosity(tf.logging.INFO)

    learning_rate = params["learning_rate"]
    # samples = features["data"]
    # timesteps = samples.shape[4]
    timesteps = 2
    samples = []
    # bands_1 = tf.constant([0,1,2,3,4])
    # bands_2 = tf.constant([5,6,7,8,9])
    # samples.append(tf.transpose(tf.nn.embedding_lookup(tf.transpose(features["data"]), bands_1)))
    # samples.append(tf.transpose(tf.nn.embedding_lookup(tf.transpose(features["data"]), bands_2)))
    samples.append(features["data"][:,:,:,0:5])
    samples.append(features["data"][:,:,:,5:10])

    height, width, _ = features["data"][0].shape

    out_encoder = []
    for i in range(timesteps):
        name_sufix = "t_" + str(i)
        out_encoder.append(unet.unet_encoder(samples[i], params, mode, name_sufix))

    with tf.variable_scope("Fusion"):
        encoded_feat = out_encoder[0]
        for i in range(1, len(out_encoder)):
            for k, feat in out_encoder[i].items():
                encoded_feat[k] = tf.concat([encoded_feat[k], feat], axis=-1, name="Fusion" + k)

    output = unet.unet_decoder(encoded_feat, params, mode)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=output)

    cropped_labels = tf.cast(layers.crop_features(labels, output.shape[1], name="labels"), tf.float32)

    loss = lossf.twoclass_cost(output, cropped_labels)

    optimizer = tf.contrib.opt.NadamOptimizer(learning_rate, name="Optimizer")
    optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

    with tf.name_scope("image_metrics"):
        input_data_vis = layers.crop_features(samples[1], output.shape[1])
        bands = tf.constant(params['bands_plot'])
        input_data_vis = tf.transpose(tf.nn.embedding_lookup(tf.transpose(input_data_vis), bands))
        input_data_vis = tf.image.convert_image_dtype(input_data_vis, tf.uint8, saturate=True)

        labels_vis = tf.image.grayscale_to_rgb(cropped_labels)

        output_vis = tf.cast(output, tf.float32)
        output_vis = tf.image.grayscale_to_rgb(output_vis)

        tf.summary.image("input_image", input_data_vis, max_outputs=params['chips_tensorboard'])
        tf.summary.image("output", output_vis, max_outputs=params['chips_tensorboard'])
        tf.summary.image("labels", labels_vis, max_outputs=params['chips_tensorboard'])

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    accuracy = tf.metrics.accuracy(labels=cropped_labels, predictions=output)
    eval_metric_ops = {"accuracy": accuracy}
    # tf.identity(accuracy, "accuracy")
    # tf.summary.scalar("accuracy", accuracy[1])

    # train_summary_hook = tf.train.SummarySaverHook(save_steps=1,
    #                                                output_dir=config.model_dir,
    #                                                summary_op=tf.summary.merge_all())

    eval_summary_hook = tf.train.SummarySaverHook(save_steps=1,
                                                  output_dir=config.model_dir + "/eval",
                                                  summary_op=tf.summary.merge_all())

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=output,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops,
                                      evaluation_hooks=[eval_summary_hook])
                                      # training_hooks=[train_summary_hook])