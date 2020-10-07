import tensorflow as tf
import numpy as np


# def twoclass_cost(predictions, labels):
def twoclass_cost(params):  # TODO: Try to set here the positive class and compute only for that.
    with tf.compat.v1.name_scope('cost'):
        predictions = tf.reshape(params['predictions'], [-1])
        trn_labels = tf.reshape(params['labels'], [-1])

        intersection = tf.reduce_sum(input_tensor=tf.multiply(predictions, trn_labels))
        union = tf.reduce_sum(input_tensor=tf.subtract(tf.add(predictions, trn_labels), tf.multiply(predictions, trn_labels)))
        loss = tf.subtract(tf.constant(1.0, dtype=tf.float32), tf.divide(intersection, union), name='loss')

        return loss


# def avg_soft_dice(logits, labels):
def avg_soft_dice(params):
    with tf.compat.v1.name_scope('cost'):
        epsilon = tf.constant(1e-6, dtype=tf.float32)
        intersection = tf.reduce_sum(input_tensor=tf.multiply(params['predictions'], params['labels_1hot']), axis=[1, 2])
        numerator = tf.multiply(tf.constant(2., dtype=tf.float32), intersection)
        denominator = tf.reduce_sum(input_tensor=tf.add(tf.square(params['predictions']), tf.square(params['labels_1hot'])), axis=[1, 2])
        dice_mean = tf.reduce_mean(input_tensor=tf.divide(numerator, tf.add(denominator, epsilon)))
        loss = tf.subtract(tf.constant(1., dtype=tf.float32), dice_mean, name='loss')
        return loss


def avg_generalized_dice(params):
    with tf.compat.v1.name_scope('cost'):
        if params['training']:
            class_weights = tf.reshape(params['class_weights']['train'], (1, params['num_classes']))
        else:
            class_weights = tf.reshape(params['class_weights']['eval'], (1, params['num_classes']))

        epsilon = tf.constant(1e-6, dtype=tf.float32)
        intersection = tf.reduce_sum(input_tensor=tf.multiply(params['predictions'], params['labels_1hot']), axis=[1, 2])
        weighted_intersection = tf.multiply(intersection, class_weights)
        numerator = tf.multiply(tf.constant(2., dtype=tf.float32), weighted_intersection)
        denominator = tf.reduce_sum(input_tensor=tf.add(params['predictions'], params['labels_1hot']), axis=[1, 2])
        denominator = tf.multiply(denominator, class_weights)
        dice_mean = tf.reduce_mean(input_tensor=tf.divide(numerator, tf.add(denominator, epsilon)))
        loss = tf.subtract(tf.constant(1., dtype=tf.float32), dice_mean, name='loss')
        return loss


# def weighted_cross_entropy(logits, labels, class_weights, num_classes, training):
def weighted_cross_entropy(params):
    num_classes = tf.shape(input=params['labels_1hot'])[-1]
    with tf.compat.v1.name_scope('cost'):
        if params['training']:
            class_weights = tf.reshape(params['class_weights']['train'], (1, num_classes))  # params['num_classes']))
        else:
            class_weights = tf.reshape(params['class_weights']['eval'], (1, num_classes))  # params['num_classes']))

        weights = tf.reduce_sum(input_tensor=tf.multiply(params['labels_1hot'], class_weights), axis=-1)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(params['labels_1hot']), logits=params['logits'])
        weighted_loss = tf.reduce_mean(input_tensor=tf.multiply(weights, loss))
        return weighted_loss


# def weighted_binary_cross_entropy(logits, labels, pos_weight):
def weighted_binary_cross_entropy(params):
    with tf.compat.v1.name_scope('cost'):
        weighted_loss = tf.nn.weighted_cross_entropy_with_logits(tf.squeeze(tf.cast(params['labels'], tf.float32)),
                                                                 tf.squeeze(params['logits']),
                                                                 params['class_weights'])
        loss = tf.reduce_mean(input_tensor=weighted_loss, name='loss')
        return loss


def unknown_loss_error():
    raise Exception('Unknown loss function!')


def parse_tfr(serialized):
    features = {'label': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                'width': tf.io.FixedLenFeature([], tf.int64, default_value=0)}

    parsed_features = tf.io.parse_single_example(serialized=serialized, features=features)
    height = parsed_features['height']
    width = parsed_features['width']

    label = tf.io.decode_raw(parsed_features['label'], tf.int32)
    label = tf.reshape(label, [height, width, 1])

    return label


def compute_weights_mean_proportion(tfrecord, classes, classes_zero=['no_data']):
    tf.compat.v1.enable_eager_execution()
    train_ds = tf.data.TFRecordDataset(tfrecord)
    train_ds = train_ds.map(parse_tfr)
    tot_count = [0] * len(classes)
    for label in train_ds:
        label = label.numpy()
        unique, count = np.unique(label, return_counts=True)
        for k, v in enumerate(unique):
            if classes[v] not in classes_zero:
                tot_count[v] += count[k]
    total = sum(tot_count)
    proportions = [i / total for i in tot_count]
    mean_prop = sum(proportions) / (len(proportions) - len(classes_zero))
    weights = [mean_prop / i if i != 0 else 0 for i in proportions]
    return weights


def compute_weights_1_minus_proportion(tfrecord, classes, classes_zero=['no_data']):
    tf.compat.v1.enable_eager_execution()
    train_ds = tf.data.TFRecordDataset(tfrecord)
    train_ds = train_ds.map(parse_tfr)
    tot_count = [0] * len(classes)
    for label in train_ds:
        label = label.numpy()
        unique, count = np.unique(label, return_counts=True)
        for k, v in enumerate(unique):
            if classes[v] not in classes_zero:
                tot_count[v] += count[k]
    total = sum(tot_count)
    proportions = [i / total for i in tot_count]
    weights = [1 - i if i != 0 else 0 for i in proportions]
    return weights


def compute_weights_inv_squared_proportion(tfrecord, classes, classes_zero=['no_data']):
    tf.compat.v1.enable_eager_execution()
    train_ds = tf.data.TFRecordDataset(tfrecord)
    train_ds = train_ds.map(parse_tfr)
    tot_count = [0] * len(classes)
    for label in train_ds:
        label = label.numpy()
        unique, count = np.unique(label, return_counts=True)
        for k, v in enumerate(unique):
            if classes[v] not in classes_zero:
                tot_count[v] += count[k]
    total = sum(tot_count)
    proportions = [i / total for i in tot_count]
    weights = [1 / (i * i) if i != 0 else 0 for i in proportions]
    return weights
