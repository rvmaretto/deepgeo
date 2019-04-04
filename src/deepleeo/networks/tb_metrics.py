import sys
import tensorflow as tf
from os import path

sys.path.insert(0, path.join(path.dirname(__file__),"../"))
import networks.layers as layers


def define_quality_metrics(labels_1hot, predictions, labels, output, loss, num_classes):
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

        metrics['mean_iou'] = tf.metrics.mean_iou(labels=labels, predictions=output,
                                                             num_classes=num_classes)
        summaries['mean_iou'] = tf.summary.scalar('mean_iou', metrics['mean_iou'][0])

        summaries['loss'] = tf.summary.scalar('loss', loss)

    return metrics, summaries

def plot_chips_tensorboard(samples, labels, output, bands_plot=[1,2,3], num_chips=2):
    with tf.name_scope("chips_predictions"):
        # print("SHAPE_LABELS: ", labels.shape)
        # print("SHAPE_SAMPLE: ", samples.shape)
        # print("SHAPE_OUTPUT: ", output.shape)
        input_data_vis = layers.crop_features(samples, output.shape[1])
        bands = tf.constant(bands_plot)
        input_data_vis = tf.transpose(tf.nn.embedding_lookup(tf.transpose(input_data_vis), bands))
        input_data_vis = tf.image.convert_image_dtype(input_data_vis, tf.uint8, saturate=True)

        labels_vis = tf.cast(labels, tf.float32)
        labels_vis = tf.image.grayscale_to_rgb(labels_vis)

        output_vis = tf.cast(output, tf.float32)
        output_vis = tf.image.grayscale_to_rgb(output_vis)

        # labels2plot_vis = tf.image.convert_image_dtype(labels2plot, tf.uint8)
        # labels2plot_vis = tf.image.grayscale_to_rgb(tf.expand_dims(labels2plot_vis, axis=-1))

        tf.summary.image("input_image", input_data_vis, max_outputs=num_chips)
        tf.summary.image("output", output_vis, max_outputs=num_chips)
        tf.summary.image("labels", labels_vis, max_outputs=num_chips)
        # tf.summary.image("labels1hot", labels2plot_vis, max_outputs=num_chips)
