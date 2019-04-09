# import tensorflow as tf
import numpy as np
import os
import sys
import math
# import skimage
# import pylab as plt
from importlib import reload
from datetime import datetime

sys.path.insert(0, '../../src')
import deepleeo.dataset.data_augment as dtaug
import deepleeo.dataset.utils as dsutils 
# import deepleeo.utils.geofunctions as gf
import deepleeo.networks.model_builder as mb

# # Load input Dataset

# In[ ]:


# DATA_DIR = os.path.join(os.path.abspath(os.path.dirname('__file__')), '../', 'data_real', 'generated')
network = 'unet'
DATA_DIR = '/home/raian/doutorado/Dados/generated'
DATASET_FILE = os.path.join(DATA_DIR, 'dataset_286x286_timesstack-2015-2016.npz')#'dataset_1.npz')

model_dir = os.path.join(DATA_DIR, 'tf_logs', 'test_%s_%s' % (network, datetime.now().strftime('%d_%m_%Y-%H_%M_%S')))
# model_dir = '/home/raian/doutorado/DeepLeEO/data_real/generated/tf_logs/test_debug'
#model_dir = os.path.join(DATA_DIR, 'tf_logs', 'test_unet_lf_17_12_2018-22_39_13')


# In[ ]:


# raster_path = os.path.join(DATA_DIR, '..', 'Landsat8_225064_17072016_R6G5B4_clip.tif')
raster_path = os.path.join(DATA_DIR, 'stacked_mosaic_2016_2017.tif')
# raster_path = os.path.join(DATA_DIR, 'stacked_mosaic_2016_2017.tif')


# In[ ]:


dataset = np.load(DATASET_FILE)

print('Data Loaded:')
print('  -> Images: ', len(dataset['images']))
print('  -> Labels: ', len(dataset['labels']))
print('  -> Classes: ', len(dataset['classes']))

print('Images shape: ', dataset['images'][0].shape, ' - DType: ', dataset['images'][0].dtype)
print('Labels shape: ', dataset['labels'][0].shape, ' - DType: ', dataset['labels'][0].dtype)
# print('UNIQUE LABELS: ', np.unique(dataset['labels']))


# # Split dataset between train, test and validation data

# In[ ]:


train_images, test_images, valid_images, train_labels, test_labels, valid_labels = dsutils.split_dataset(dataset)

print('Splitted dataset:')
print('  -> Train images: ', train_images.shape)
print('  -> Test images: ', test_images.shape)
print('  -> Validation images: ', valid_images.shape)
print('  -> Train Labels: ', train_labels.shape)
print('  -> Test Labels: ', test_labels.shape)
print('  -> Validation Labels: ', valid_labels.shape)


# # Perform Data Augmentation

angles = [90, 180, 270]
rotated_imgs = dtaug.rotate_images(train_images, angles)
flipped_imgs = dtaug.flip_images(train_images)

train_images = np.concatenate((train_images, rotated_imgs))
train_images = np.concatenate((train_images, flipped_imgs))

rotated_lbls = dtaug.rotate_images(train_labels, angles)
flipped_lbls = dtaug.flip_images(train_labels)

train_labels = np.concatenate((train_labels, rotated_lbls))
train_labels = np.concatenate((train_labels, flipped_lbls)).astype(dtype=np.int32)

print('Data Augmentation Applied:')
print('  -> Train Images: ', train_images.shape)
print('  -> Train Labels: ', train_labels.shape)
print('  -> Test Images: ', test_images.shape)
print('  -> Test Labels: ', test_labels.shape)


values, count = np.unique(train_labels, return_counts=True)
print('Class Values: ', values, '  - Count: ', count)
print('Class Names: ', dataset['classes'])

defor_proportion = count[1] / (count[0] + count[1])
non_defor_proportion = count[0] / (count[0] + count[1])
print('Defining weights for classes:')
print('  -> Deforestation Proportion: ', defor_proportion)
print('  -> Non deforestation Proportion: ', non_defor_proportion)

print('Ratio: ', non_defor_proportion / defor_proportion)

mean_proportion = (defor_proportion + non_defor_proportion) / 2
print('  -> Median Proportion: ', mean_proportion)
weight_defor = mean_proportion / defor_proportion
weight_non_defor = mean_proportion / non_defor_proportion

print('  -> Weights: [', weight_non_defor, ', ', weight_defor, ']')


# # Train the Network

# In[ ]:


# # Train the Network
params = {
    'epochs': 100,
    'batch_size': 100,
    'learning_rate': 0.1,
    'learning_rate_decay': True,
    'decay_rate': 0.95,
    'decay_steps': 260,
    'l2_reg_rate': 0.5,
    'var_scale_factor': 2.0,
    'chips_tensorboard': 2,
    'dropout_rate': 0.5,
    'fusion': 'early',
    'loss_func': 'weighted_crossentropy',
    'class_weights': weight_defor,#[weight_non_defor, weight_defor],
    'num_classes': len(dataset['classes']),
    'num_compositions': 2,
    'bands_plot': [[1, 2, 3], [6, 7, 8]]
}


# In[ ]:


reload(mb)
model = mb.ModelBuilder(network)
model.train(train_images, test_images, train_labels, test_labels, params, model_dir)
