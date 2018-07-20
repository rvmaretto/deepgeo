import tensorflow as tf

def softmax_loss_cross_entropy(net_score, labels, num_classes, weight_classes=None):
    with tf.name_scope("Loss"):
        #TODO: Can I use this one? What is the difference. Can it operate pixelwise??
        #tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=up_score)
        up_score = tf.reshape(net_score, (-1, num_classes))
        epsilon = tf.constant(value=1e-4)
        labels = tf.to_float(tf.reshape(labels, -1, num_classes))

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