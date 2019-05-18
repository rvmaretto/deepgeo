import tensorflow as tf


def _rot90(image, label):
    image = tf.image.rot90(image, 1)
    label = tf.image.rot90(label, 1)
    return image, label


def _rot180(image, label):
    image = tf.image.rot90(image, 2)
    label = tf.image.rot90(label, 2)
    return image, label


def _rot270(image, label):
    image = tf.image.rot90(image, 3)
    label = tf.image.rot90(label, 3)
    return image, label


def _flip_left_right(image, label):
    image = tf.image.flip_left_right(image)
    label = tf.image.flip_left_right(label)
    return image, label


def _flip_up_down(image, label):
    image = tf.image.flip_up_down(image)
    label = tf.image.flip_up_down(label)
    return image, label


def _flip_transpose(image, label):
    image = tf.image.transpose_image(image)
    label = tf.image.transpose_image(label)
    return image, label


class DatasetLoader(object):
    data_aug_operations = {'rot90': _rot90,
                           'rot180': _rot180,
                           'rot270': _rot270,
                           'flip_left_right': _flip_left_right,
                           'flip_up_down': _flip_up_down,
                           'flip_transpose': _flip_transpose}

    features = {'image': tf.FixedLenFeature([], tf.string, default_value=''),
                'channels': tf.FixedLenFeature([], tf.int64, default_value=0),
                'label': tf.FixedLenFeature([], tf.string, default_value=''),
                'height': tf.FixedLenFeature([], tf.int64, default_value=0),
                'width': tf.FixedLenFeature([], tf.int64, default_value=0)}

    def __init__(self, train_dataset):
        self.dataset = train_dataset

    def set_tfrecord_features(self, features):
        self.features = features

    def get_tfrecord_features(self):
        return self.features

    def get_image_shape(self):
        for record in tf.python_io.tf_record_iterator(self.dataset):
            self.chip_shape = tf.train.Example()
            self.chip_shape.ParseFromString(record)
            break
        return self.chip_shape

    def get_dataset_size(self):
        number_of_chips = 0
        for record in tf.python_io.tf_record_iterator(self.dataset):
            number_of_chips += 1
            if number_of_chips == 1:
                chip_shape = tf.train.Example()
                chip_shape.ParseFromString(record)
        return number_of_chips

    def _parse_shape(self, serialized):
        features = self.features
        parsed_features = tf.parse_single_example(serialized=serialized, features=features)
        num_bands = parsed_features['channels']
        height = parsed_features['height']
        width = parsed_features['width']
        return [num_bands, height, width]

    def _parse_function(self, serialized):
        parsed_features = tf.parse_single_example(serialized=serialized, features=self.featuresfeatures)
        num_bands = parsed_features['channels']
        height = parsed_features['height']
        width = parsed_features['width']

        #shape_img = tf.stack([height, width, 10])
        #shape_lbl = tf.stack([height, width, 1])
        shape_img = [286, 286, 10]
        shape_lbl = [286, 286, 1]

        image = tf.decode_raw(parsed_features['image'], tf.float32)
        image = tf.reshape(image, shape_img)

        label = tf.decode_raw(parsed_features['label'], tf.int32)
        label = tf.reshape(label, shape_lbl)
        return image, label

    def tfrecord_input_fn(self, params, train=True):
        dataset = tf.data.TFRecordDataset(self.dataset)
        train_input = dataset.map(self._parse_function, num_parallel_calls=40)
        if train:
            aug_datasets = []
            for op in params['data_aug_ops']:
                aug_ds = train_input.map(self.data_aug_operations[op], num_parallel_calls=40)
                aug_datasets.append(aug_ds)

            for ds in aug_datasets:
                train_input = train_input.concatenate(ds)

            train_input = train_input.shuffle(10000)
            train_input = train_input.repeat(params['epochs'])
        else:
            train_input.repeat(1)
        train_input = train_input.batch(params['batch_size'])
        train_input = train_input.prefetch(1000)
        return train_input

    # TODO: Implement this method.
    # def register_dtaug_op
