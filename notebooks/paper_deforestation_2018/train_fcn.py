import numpy as np
import os
import sys
from importlib import reload
from datetime import datetime
import tensorflow as tf

sys.path.insert(0, '../../src')
import deepgeo.networks.model_builder as mb

# # Load input Dataset

# In[ ]:

network = 'unet'
DATA_DIR = '/home/raian/doutorado/Dados/generated'

class_names = ['no_data', 'not_deforestation', 'deforestation']

DATASET = os.path.join(DATA_DIR, 'dataset_286x286_tmstk-2013-2017')
train_tfrecord = os.path.join(DATASET, 'dataset_train.tfrecord')
test_tfrecord = os.path.join(DATASET, 'dataset_test.tfrecord')
val_dataset = os.path.join(DATASET, 'dataset_valid.npz')
# val_tfrecord = os.path.join(DATASET, 'dataset_validation.tfrecord')

model_dir = os.path.join(DATA_DIR, 'tf_logs', 'experiments', network,
                         'test_%s_%s' % (network, datetime.now().strftime('%d_%m_%Y-%H_%M_%S')))


def parse(serialized):
    features = {'label': tf.FixedLenFeature([], tf.string, default_value=''),
                'height': tf.FixedLenFeature([], tf.int64, default_value=0),
                'width': tf.FixedLenFeature([], tf.int64, default_value=0)}

    parsed_features = tf.parse_single_example(serialized=serialized, features=features)
    height = parsed_features['height']
    width = parsed_features['width']

    label = tf.decode_raw(parsed_features['label'], tf.int32)
    label = tf.reshape(label, [height, width, 1])

    return label


def compute_weights_mean_proportion(tfrecord, classes, classes_zero=['no_data']):
    tf.enable_eager_execution()
    train_ds = tf.data.TFRecordDataset(tfrecord)
    train_ds = train_ds.map(parse)
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


weights_train = compute_weights_mean_proportion(train_tfrecord, class_names, ['no_data'])
weights_eval = compute_weights_mean_proportion(test_tfrecord, class_names, ['no_data'])


# Train the Network

params = {
    'epochs': 100,
    'batch_size': 40,
    'chip_size': 286,
    'bands': 10,
    'filter_reduction': 0.5,
    'learning_rate': 0.1,
    'learning_rate_decay': True,
    'decay_rate': 0.95,
    'l2_reg_rate': 0.0005,
    # 'var_scale_factor': 2.0,  # TODO: Put the initializer as parameter
    'chips_tensorboard': 2,
    # 'dropout_rate': 0.5,  # TODO: Put a bool parameter to apply or not Dropout
    'fusion': 'early',
    'loss_func': 'weighted_crossentropy',
    'data_aug_ops': ['rot90', 'rot180', 'rot270', 'flip_left_right',
                     'flip_up_down', 'flip_transpose'],
    'class_weights': {'train': weights_train, 'eval': weights_eval},
    'num_classes': len(class_names),
    'class_names': ['no data', 'not deforestation', 'deforestation'],
    'num_compositions': 2,
    'bands_plot': [[1, 2, 3], [6, 7, 8]],
    'Notes': 'Migrating to TFRecord and MirroredStrategy. All Data Augmentation!'
}


reload(mb)
model = mb.ModelBuilder(network)
model.train(train_tfrecord, test_tfrecord, params, model_dir)
dataset = np.load(val_dataset)
model.validate(dataset['chips'], dataset['labels'], params, model_dir)
# model.validate(val_tfrecord, params, model_dir)
