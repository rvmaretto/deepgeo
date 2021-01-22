import sys
import tensorflow as tf
from os import path
import tensorflow_addons as tfa

sys.path.insert(0, path.join(path.dirname(__file__), '../'))
import networks.layers as layers

def f1_score(labels, predictions):
    return 2 * (tf.compat.v1.metrics.recall(labels, predictions) *
                tf.compat.v1.metrics.precision(labels, predictions)) / \
           (tf.compat.v1.metrics.recall(labels, predictions) + tf.compat.v1.metrics.precision(labels, predictions))


def define_quality_metrics(labels_1hot, predictions, logits, labels, output, loss, params):
    metrics = {}
    summaries = {}
    with tf.compat.v1.name_scope('quality_metrics'):
        # metrics['f1_score'] = f1_score(labels=labels_1hot, predictions=predictions)
        # summaries['f1_score'] = tf.compat.v1.summary.scalar('f1-score', metrics['f1_score'][1])

        metrics['accuracy'] = tf.compat.v1.metrics.accuracy(labels=labels, predictions=output)
        summaries['accuracy'] = tf.compat.v1.summary.scalar('accuracy', metrics['accuracy'][1])

        cross_entropy = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=labels_1hot, logits=predictions)
        metrics['cross_entropy'] = tf.compat.v1.metrics.mean(cross_entropy)
        summaries['cross_entropy'] = tf.compat.v1.summary.scalar('cross_entropy', metrics['cross_entropy'][1])

        metrics['auc-roc'] = tf.compat.v1.metrics.auc(labels=labels_1hot, predictions=predictions)
        summaries['auc-roc'] = tf.compat.v1.summary.scalar('auc_roc', metrics['auc-roc'][1])

        # metrics['mean_iou'] = tf.metrics.mean_iou(labels=labels, predictions=predictions,
        #                                                      num_classes=params['num_classes'])
        # summaries['mean_iou'] = tf.summary.scalar('mean_iou', metrics['mean_iou'][0])

        summaries['loss'] = tf.compat.v1.summary.scalar('loss', loss)

    return metrics, summaries


def plot_chips_tensorboard(samples, labels, output, params):
    with tf.compat.v1.name_scope("input_chips"):
        input_data = layers.crop_features(samples, output.shape[1])
        plots = []
        if 'num_compositions' not in params:
            params['num_compositions'] = 1
        if not isinstance(params['bands_plot'][0], list):
            params['bands_plot'] = [params['bands_plot']]
        for i in range(0, params['num_compositions']):
            bands = tf.constant(params['bands_plot'][i])
            input_data_vis = tf.transpose(a=tf.nn.embedding_lookup(params=tf.transpose(a=input_data), ids=bands))
            input_data_vis = tf.image.convert_image_dtype(input_data_vis, tf.uint8, saturate=True)
            tf.compat.v1.summary.image("input_image_c{}".format(i), input_data_vis, max_outputs=params['chips_tensorboard'])

    with tf.compat.v1.name_scope("labels"):
        labels_vis = tf.cast(labels, tf.float32)
        labels_vis = tf.image.grayscale_to_rgb(labels_vis)
        tf.compat.v1.summary.image("labels", labels_vis, max_outputs=params['chips_tensorboard'])

        output_vis = tf.cast(output, tf.float32)
        output_vis = tf.image.grayscale_to_rgb(output_vis)
        tf.compat.v1.summary.image("output", output_vis, max_outputs=params['chips_tensorboard'])

        # labels2plot_vis = tf.image.convert_image_dtype(labels2plot, tf.uint8)
        # labels2plot_vis = tf.image.grayscale_to_rgb(tf.expand_dims(labels2plot_vis, axis=-1))
        # tf.summary.image("labels1hot", labels2plot_vis, max_outputs=num_chips)
