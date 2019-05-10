import sys
import tensorflow as tf
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), '../'))
import networks.layers as layers


def define_quality_metrics(labels_1hot, predictions, logits, labels, output, loss, params):
    metrics = {}
    summaries = {}
    with tf.name_scope('quality_metrics'):
        metrics['f1_score'] = tf.contrib.metrics.f1_score(labels=labels_1hot, predictions=predictions)
        summaries['f1_score'] = tf.summary.scalar('f1-score', metrics['f1_score'][1])

        metrics['accuracy'] = tf.metrics.accuracy(labels=labels, predictions=output)
        summaries['accuracy'] = tf.summary.scalar('accuracy', metrics['accuracy'][1])

        cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels_1hot, logits=predictions)
        metrics['cross_entropy'] = tf.metrics.mean(cross_entropy)
        summaries['cross_entropy'] = tf.summary.scalar('cross_entropy', metrics['cross_entropy'][1])

        metrics['auc-roc'] = tf.metrics.auc(labels=labels, predictions=predictions)

        # metrics['mean_iou'] = tf.metrics.mean_iou(labels=labels, predictions=predictions,
        #                                                      num_classes=params['num_classes'])
        # summaries['mean_iou'] = tf.summary.scalar('mean_iou', metrics['mean_iou'][0])

        summaries['loss'] = tf.summary.scalar('loss', loss)

    return metrics, summaries


def plot_chips_tensorboard(samples, labels, output, params):
    with tf.name_scope("input_chips"):
        input_data = layers.crop_features(samples, output.shape[1])
        plots = []
        for i in range(0, params['num_compositions']):
            bands = tf.constant(params['bands_plot'][i])
            input_data_vis = tf.transpose(tf.nn.embedding_lookup(tf.transpose(input_data), bands))
            input_data_vis = tf.image.convert_image_dtype(input_data_vis, tf.uint8, saturate=True)
            tf.summary.image("input_image_c{}".format(i), input_data_vis, max_outputs=params['chips_tensorboard'])

    with tf.name_scope("labels"):
        labels_vis = tf.cast(labels, tf.float32)
        labels_vis = tf.image.grayscale_to_rgb(labels_vis)
        tf.summary.image("labels", labels_vis, max_outputs=params['chips_tensorboard'])

        output_vis = tf.cast(output, tf.float32)
        output_vis = tf.image.grayscale_to_rgb(output_vis)
        tf.summary.image("output", output_vis, max_outputs=params['chips_tensorboard'])

        # labels2plot_vis = tf.image.convert_image_dtype(labels2plot, tf.uint8)
        # labels2plot_vis = tf.image.grayscale_to_rgb(tf.expand_dims(labels2plot_vis, axis=-1))
        # tf.summary.image("labels1hot", labels2plot_vis, max_outputs=num_chips)
