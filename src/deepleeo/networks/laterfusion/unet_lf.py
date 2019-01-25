import sys
from os import path
import tensorflow as tf

sys.path.insert(0, path.join(path.dirname(__file__), ".."))
import networks.unet as unet
import networks.layers as layers
import networks.loss_functions as lossf
import networks.tb_metrics as tbm

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

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=output,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops,
                                      evaluation_hooks=[eval_summary_hook])
                                      # training_hooks=[train_summary_hook])