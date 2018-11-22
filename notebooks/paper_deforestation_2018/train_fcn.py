
# coding: utf-8

# In[2]:


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
import deepleeo.networks.model_builder as mb
# from deepleeo.networks import fcn8s

reload(dtaug)
reload(dsutils)
# reload(fcn)
reload(mb)
reload(gf)


# # Load input Dataset

# In[3]:


# DATA_DIR = os.path.join(os.path.abspath(os.path.dirname("__file__")), '../', 'data_real', 'generated')
# DATASET_FILE = os.path.join(DATA_DIR, 'samples_dataset_bin.npz')
network = "fcn8s"
DATA_DIR = "/home/raian/doutorado/Dados/generated"
DATASET_FILE = os.path.join(DATA_DIR, 'dataset_286x286_timesstack-2015-2016.npz')#'dataset_1.npz')
#TODO: Put network name here in the path
model_dir = os.path.join(DATA_DIR, 'tf_logs', "test_%s_%s" % (network, datetime.now().strftime('%d_%m_%Y-%H_%M_%S')))
# model_dir = "/home/raian/doutorado/DeepLeEO/data_real/generated/tf_logs/test_debug"


# In[4]:


# raster_path = os.path.join(DATA_DIR, "..", "Landsat8_225064_17072016_R6G5B4_clip.tif")
raster_path = os.path.join(DATA_DIR, "..", "mosaic_2016.tif")


# In[5]:


dataset = np.load(DATASET_FILE)

print("Data Loaded:")
print("  -> Images: ", len(dataset["images"]))
print("  -> Labels: ", len(dataset["labels"]))
print("  -> Classes: ", len(dataset["classes"]))

print("Images shape: ", dataset["images"][0].shape, " - DType: ", dataset["images"][0].dtype)
print("Labels shape: ", dataset["labels"][0].shape, " - DType: ", dataset["labels"][0].dtype)
# print("UNIQUE LABELS: ", np.unique(dataset["labels"]))


# In[6]:


#plt.figure(figsize=(4,4))
#img_plt = skimage.img_as_float(dataset["images"][0])
#img_plt = dataset["images"][0]
#plt.imshow(img_plt)
#plt.axis('off')


# # Perform Data Augmentation

# In[7]:


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

#print("  -> Unique Images: ", np.unique(new_dataset["images"]))
# print("  -> Unique Labels: ", np.unique(new_dataset["labels"]))


# # Split dataset between train, test and validation data

# In[8]:


train_images, test_images, valid_images, train_labels, test_labels, valid_labels = dsutils.split_dataset(dataset)

print("Splitted dataset:")
print("  -> Train images: ", train_images.shape)
print("  -> Test images: ", test_images.shape)
print("  -> Validation images: ", valid_images.shape)
print("  -> Train Labels: ", train_labels.shape)
print("  -> Test Labels: ", test_labels.shape)
print("  -> Validation Labels: ", valid_labels.shape)

# print("  -> UNIQUE TRAIN LABELS: ", np.unique(train_labels), " - Type: ", train_labels.dtype)
# print("  -> UNIQUE TEST LABELS: ", np.unique(test_labels), " - Type: ", test_labels.dtype)
# print("  -> UNIQUE VALIDATION LABELS: ", np.unique(valid_labels))


# # Train the Network

# In[9]:


params = {
    "epochs": 600,
    "batch_size": 200,
    "learning_rate": 0.0001,
    "class_names": dataset["classes"],
    "multi_gpu": False
}


# In[ ]:


# reload(fcn)
reload(mb)

model = mb.ModelBuilder(network)
model.train(train_images, test_images, train_labels, test_labels, params, model_dir)
