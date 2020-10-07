import tensorflow as tf

# The code in this file is based on: https://stackoverflow.com/questions/52266000/avoiding-tf-data-dataset-from-tensor-slices-with-estimator-api
class IteratorInitializerHook(tf.estimator.SessionRunHook):
    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None # Will be set in the input_fn

    def after_create_session(self, session, coord):
        print("INITIALIZE")
        # Initialize the iterator with the data feed_dict
        self.iterator_initializer_func(session)

def get_input_fn(imgs, labels, batch_size, shuffle=True):
    iterator_initializer_hook = IteratorInitializerHook()

    def input_fn():
        features_placeholder = tf.compat.v1.placeholder(imgs.dtype, imgs.shape)
        labels_placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)

        dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
        if shuffle:
            buffer_size = imgs.shape[0] + 100
            dataset = dataset.shuffle(buffer_size).repeat().batch(batch_size)
        else:
            dataset = dataset.repeat().batch(batch_size)

        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        next_example, next_label = iterator.get_next()

        iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(iterator.initializer, feed_dict={features_placeholder: imgs, labels_placeholder: labels})

        return dataset#next_example, next_label

    return input_fn, iterator_initializer_hook