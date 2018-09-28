
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import os
import sys
import skimage
import pylab as plt
from importlib import reload
from datetime import datetime

sys.path.insert(0, '../src')
import deepleeo.dataset.data_augment as dtaug
import deepleeo.dataset.utils as dsutils 
import deepleeo.utils.geofunctions as gf
from deepleeo.networks import fcn

reload(dtaug)
reload(dsutils)
reload(fcn)
reload(gf)

# DATA_DIR = os.path.join(os.path.abspath(os.path.dirname("__file__")), '../', 'data_real', 'generated')
# DATASET_FILE = os.path.join(DATA_DIR, 'samples_dataset_bin.npz')
DATA_DIR = "/home/raian/doutorado/Dados/generated"
DATASET_FILE = os.path.join(DATA_DIR, 'dataset_286x286_2016.npz')#'dataset_1.npz')
model_dir = os.path.join(DATA_DIR, 'tf_logs', "test_FCN_%s" % datetime.now().strftime('%d_%m_%Y-%H_%M_%S'))
# model_dir = "/home/raian/doutorado/DeepLeEO/data_real/generated/tf_logs/test_debug"


# raster_path = os.path.join(DATA_DIR, "..", "Landsat8_225064_17072016_R6G5B4_clip.tif")
raster_path = os.path.join(DATA_DIR, "..", "mosaic_2016.tif")

dataset = np.load(DATASET_FILE)

print("Data Loaded:")
print("  -> Images: ", len(dataset["images"]))
print("  -> Labels: ", len(dataset["labels"]))
print("  -> Classes: ", len(dataset["classes"]))

print("Images shape: ", dataset["images"][0].shape, " - DType: ", dataset["images"][0].dtype)
print("Labels shape: ", dataset["labels"][0].shape, " - DType: ", dataset["labels"][0].dtype)
print("UNIQUE LABELS: ", np.unique(dataset["labels"]))


reload(dtaug)
angles = [90, 180, 270]
rotated_imgs = dtaug.rotate_images(dataset["images"], angles)
flipped_imgs = dtaug.flip_images(dataset["images"])

new_dataset = {}
new_dataset["images"] = np.concatenate((dataset["images"], rotated_imgs))
new_dataset["images"] = np.concatenate((new_dataset["images"], flipped_imgs))

rotated_lbls = dtaug.rotate_images(dataset["labels"], angles)
flipped_lbls = dtaug.flip_images(dataset["labels"])

new_dataset["labels"] = np.concatenate((dataset["labels"], rotated_lbls))
new_dataset["labels"] = np.concatenate((new_dataset["labels"], flipped_lbls)).astype(dtype=np.int32)

new_dataset["classes"] = dataset["classes"]

print("Data Augmentation Applied:")
print("  -> Images: ", new_dataset["images"].shape)
print("  -> Labels: ", new_dataset["labels"].shape)


train_images, test_images, valid_images, train_labels, test_labels, valid_labels = dsutils.split_dataset(new_dataset)

print("Splitted dataset:")
print("  -> Train images: ", train_images.shape)
print("  -> Test images: ", test_images.shape)
print("  -> Validation images: ", valid_images.shape)
print("  -> Train Labels: ", train_labels.shape)
print("  -> Test Labels: ", test_labels.shape)
print("  -> Validation Labels: ", valid_labels.shape)

params = {
    "epochs": 200,
    "batch_size": 200,
    "learning_rate": 0.0001,
    "class_names": dataset["classes"],
    "multi_gpu": False
}

fcn.fcn_train(train_images, test_images, train_labels, test_labels, params, model_dir)
