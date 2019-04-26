import tensorflow as tf
import numpy as np
import os
import sys
from importlib import reload
from shutil import rmtree

sys.path.insert(0, '../../src')
import deepgeo.dataset.data_augment as dtaug
import deepgeo.dataset.utils as dsutils
import deepgeo.networks.model_builder as mb
import deepgeo.common.filesystem as fs
import deepgeo.common.geofunctions as gf

reload(dtaug)
reload(dsutils)
reload(mb)
reload(gf)

current_path = os.path.abspath(os.path.dirname('__file__'))
DATA_DIR = os.path.join(current_path, '..', '..', 'data_real', 'generated')
network = 'fcn8s'
DATASET_FILE = os.path.join(DATA_DIR, 'samples_dataset_bin.npz')

# model_dir = os.path.join(DATA_DIR, 'tf_logs', 'test_%s_%s' % (network, datetime.now().strftime('%d_%m_%Y-%H_%M_%S')))
model_dir = os.path.join(DATA_DIR, 'tf_logs', 'test_debug')

fs.mkdir(model_dir)

if os.listdir(model_dir):
    for fname in os.listdir(model_dir):
        fpath = os.path.join(model_dir, fname)
        try:
            if os.path.isfile(fpath):
                os.unlink(fpath)
            else:
                fs.delete_dir(fpath)
        except Exception as e:
            print(e)

dataset = np.load(DATASET_FILE)

print('Data Loaded:')
print('  -> Images: ', len(dataset['images']))
print('  -> Labels: ', len(dataset['labels']))
print('  -> Classes: ', len(dataset['classes']))

print('Images shape: ', dataset['images'][0].shape, ' - DType: ', dataset['images'][0].dtype)
print('Labels shape: ', dataset['labels'][0].shape, ' - DType: ', dataset['labels'][0].dtype)
# print('UNIQUE LABELS: ', np.unique(dataset['labels']))

## Split dataset between train, test and validation data
train_images, test_images, valid_images, train_labels, test_labels, valid_labels = dsutils.split_dataset(dataset, perc_test=20)

print('Splitted dataset:')
print('  -> Train images: ', train_images.shape)
print('  -> Test images: ', test_images.shape)
print('  -> Validation images: ', valid_images.shape)
print('  -> Train Labels: ', train_labels.shape)
print('  -> Test Labels: ', test_labels.shape)
print('  -> Validation Labels: ', valid_labels.shape)

def compute_weights_mean_proportion(batch_array, classes, classes_zero=['no_data']):
    values, count = np.unique(batch_array, return_counts=True)
    count = [count[i] if classes[i] not in classes_zero else 0 for i in range(0, len(count))]
    total = sum(count)
    proportions = [i / total for i in count]
    mean_prop = sum(proportions)/ (len(proportions) - len(classes_zero))
    weights = [mean_prop / i if i != 0 else 0 for i in proportions]
    return weights

weights_train = compute_weights_mean_proportion(train_labels, dataset['classes'])
weights_eval = compute_weights_mean_proportion(test_labels, dataset['classes'])

print(weights_train)
print(weights_eval)


# # Train the Network
params = {
    'epochs': 5,
    'batch_size': 8,
    'learning_rate': 0.1,
    'learning_rate_decay': True,
    'decay_rate': 0.95,
    # 'decay_steps': 1286,
    'l2_reg_rate': 0.0005,
    # 'var_scale_factor': 2.0,  # TODO: Put the initializer as parameter
    'chips_tensorboard': 2,
    'dropout_rate': 0.5,  # TODO: Put a bool parameter to apply or not Dropout
    'fusion': 'late',
    'loss_func': 'weighted_crossentropy',
    'class_weights': {'train': weights_train, 'eval': weights_eval},
    'num_classes': len(dataset['classes']),
    'num_compositions': 2,
     'bands_plot': [[1, 2, 3], [0, 1, 2]]
}

model = mb.ModelBuilder(network)
model.train(train_images, test_images, train_labels, test_labels, params, model_dir)
