import tensorflow as tf


def twoclass_cost(predictions, labels):
    with tf.name_scope('cost'):
        predictions = tf.reshape(predictions, [-1])
        trn_labels = tf.reshape(labels, [-1])

        intersection = tf.reduce_sum(tf.multiply(predictions, trn_labels))
        union = tf.reduce_sum(tf.subtract(tf.add(predictions, trn_labels), tf.multiply(predictions, trn_labels)))
        loss = tf.subtract(tf.constant(1.0, dtype=tf.float32), tf.divide(intersection,union), name='loss')

        return loss


def avg_soft_dice(logits, labels):
    with tf.name_scope('cost'):
        epsilon = tf.constant(1e-6, dtype=tf.float32)
        intersection = tf.reduce_sum(tf.multiply(logits, labels), axis=[1, 2])
        numerator = tf.multiply(tf.constant(2., dtype=tf.float32), intersection)
        denominator = tf.reduce_sum(tf.add(tf.square(logits), tf.square(labels)), axis=[1, 2])
        dice_mean = tf.reduce_mean(tf.divide(numerator, tf.add(denominator, epsilon)))
        loss = tf.subtract(tf.constant(1., dtype=tf.float32), dice_mean, name='loss')
        # tf.add_to_collection('loss', loss)
        # loss = tf.add_n(tf.get_collection('loss'), name='loss')
        return loss


def weighted_cross_entropy(logits, labels, class_weights, num_classes, training):
    with tf.name_scope('cost'):
        if training:
            class_weights = tf.reshape(class_weights['train'], (1, num_classes))
        else:
            class_weights = tf.reshape(class_weights['eval'], (1, num_classes))

        weights = tf.reduce_sum(tf.multiply(labels, class_weights), axis=-1)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        weighted_loss = tf.reduce_mean(tf.multiply(weights, loss))
        return weighted_loss


def weighted_binary_cross_entropy(logits, labels, pos_weight):
    with tf.name_scope('cost'):
        weighted_loss = tf.nn.weighted_cross_entropy_with_logits(tf.squeeze(tf.cast(labels, tf.float32)),
                                                                 tf.squeeze(logits),
                                                                 pos_weight)
        loss = tf.reduce_mean(weighted_loss, name='loss')
        return loss


def unknown_loss_error():
    raise Exception('Unknown loss function!')