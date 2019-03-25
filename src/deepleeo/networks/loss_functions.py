import tensorflow as tf


def twoclass_cost(predictions, labels):
    with tf.name_scope('cost'):
        predictions = tf.reshape(predictions, [-1])
        trn_labels = tf.reshape(labels, [-1])

        intersection = tf.reduce_sum( tf.multiply(predictions,trn_labels) )
        union = tf.reduce_sum( tf.subtract( tf.add(predictions,trn_labels) , tf.multiply(predictions,trn_labels) ) )
        loss = tf.subtract( tf.constant(1.0, dtype=tf.float32), tf.divide(intersection,union), name='loss')

        return loss


def inverse_mean_iou(predictions, labels, num_classes):
    with tf.name_scope('cost'):
        mean_iou, conf_mat = tf.metrics.mean_iou(labels=labels, predictions=predictions, num_classes=num_classes)
        return tf.cast(tf.subtract(1.0, mean_iou), tf.float64)


# def avg_soft_dice(predictions, labels, num_classes):
#     with tf.name_scope('cost'):
#         avg_dice = tf.constant(0)
#         for i in range(num_classes):
#             intersection = tf.multiply(predictions, labels)
#             pred_sum = tf.pow(tf.reduce_sum(predictions), 2)
#             labels_sum = tf.pow(tf.reduce_sum(predictions), 2)
#             numerator = 2 * tf.reduce_sum(intersection)
#             denominator = pred_sum + labels_sum
#             soft_dice = 1 - tf.divide(numerator, denominator)
#             avg_dice += soft_dice
#
#         return avg_dice / num_classes

def avg_soft_dice(predictions, labels):
    with tf.name_scope('cost'):
        epsilon = tf.constant(1e-6, dtype=tf.float32)
        intersection = tf.reduce_sum(tf.multiply(predictions, labels), axis=[1, 2])
        numerator = tf.multiply(tf.constant(2., dtype=tf.float32), intersection)
        denominator = tf.reduce_sum(tf.square(predictions) + tf.square(labels), axis=[1, 2])
        dice_mean = tf.reduce_mean(tf.divide(numerator, tf.add(denominator, epsilon)))
        return tf.subtract(tf.constant(1., dtype=tf.float32), dice_mean, name='loss')
