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
    # data_aug_operations = {
    #     'rot90': _rot90,
    #     'rot180': _rot180,
    #     'rot270': _rot270,
    #     'flip_left_right': _flip_left_right,
    # }

    def _parse_function(self, serialized):
        features = {'image': tf.FixedLenFeature([], tf.string, default_value=''),
                    'channels': tf.FixedLenFeature([], tf.int64, default_value=0),
                    'label': tf.FixedLenFeature([], tf.string, default_value=''),
                    'height': tf.FixedLenFeature([], tf.int64, default_value=0),
                    'width': tf.FixedLenFeature([], tf.int64, default_value=0)}

        parsed_features = tf.parse_single_example(serialized=serialized, features=features)
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

    def tfrecord_input_fn(self, train_dataset, params, train=True):
        dataset = tf.data.TFRecordDataset(train_dataset)
        train_input = dataset.map(self._parse_function, num_parallel_calls=40)
        dt_augs = [_rot90, _rot180, _rot270, _flip_left_right, _flip_up_down, _flip_transpose]
        if train:
            aug_datasets = []
            for op in dt_augs:
                aug_ds = train_input.map(op, num_parallel_calls=40)
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
