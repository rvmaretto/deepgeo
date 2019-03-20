import tensorflow as tf

def softmax_loss_cross_entropy(net_score, labels, num_classes, weight_classes=None):
    with tf.name_scope("Loss"):
        #TODO: Can I use this one? What is the difference. Can it operate pixelwise??
        #tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=up_score)
        up_score = tf.reshape(net_score, (-1, num_classes))
        epsilon = tf.constant(value=1e-4)
        labels = tf.reshape(labels, -1, num_classes)

        softmax = tf.nn.softmax(up_score) + epsilon

        if(weight_classes is None):
            cross_entropy = tf.reduce_sum(tf.multiply(labels * tf.log(softmax)),
                                          reduction_indices=[1], name="cross_entropy")
        else:
            cross_entropy = tf.reduce_sum(tf.multiply(labels * tf.log(softmax), weight_classes),
                                          reduction_indices=[1], name="cross_entropy")

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name="xEntropy_mean")
        tf.add_to_collection("losses", cross_entropy_mean)
        loss = tf.add_n(tf.get_collection("losses"), name="total_loss")

    return loss

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

def inverse_f1_score(predictions, labels):
    with tf.name_scope('cost'):
        f1_score = tf.contrib.metrics.f1_score(labels=labels, predictions=predictions)
        return (tf.subtract(1.0, f1_score[0]), f1_score[1])
