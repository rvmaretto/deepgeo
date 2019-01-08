
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
import os
import sys
from importlib import reload
from datetime import datetime

sys.path.insert(0, '../../src')
import deepleeo.dataset.data_augment as dtaug
import deepleeo.dataset.utils as dsutils 
import deepleeo.utils.geofunctions as gf
import deepleeo.networks.model_builder as mb

reload(dtaug)
reload(dsutils)
reload(mb)
reload(gf)


# # Load input Dataset

# In[3]:


# DATA_DIR = os.path.join(os.path.abspath(os.path.dirname("__file__")), '../', 'data_real', 'generated')
network = "fcn8s"
DATA_DIR = "/home/raian/doutorado/Dados/generated/igarss"
DATASET_FILE = os.path.join(DATA_DIR, 'samples_dataset_igarss.npz')#'dataset_1.npz')

model_dir = os.path.join(DATA_DIR, 'tf_logs', "test_%s_%s" % (network, datetime.now().strftime('%d_%m_%Y-%H_%M_%S')))


# In[4]:


dataset = np.load(DATASET_FILE)

print("Data Loaded:")
print("  -> Images: ", len(dataset["images"]))
print("  -> Labels: ", len(dataset["labels"]))
print("  -> Classes: ", len(dataset["classes"]))

print("Images shape: ", dataset["images"][0].shape, " - DType: ", dataset["images"][0].dtype)
print("Labels shape: ", dataset["labels"][0].shape, " - DType: ", dataset["labels"][0].dtype)
# print("UNIQUE LABELS: ", np.unique(dataset["labels"]))


# # Perform Data Augmentation

# In[ ]:


new_dataset = {}
new_dataset["images"] = dataset["images"]
new_dataset["labels"] = dataset["labels"]

angles = [90, 180, 270]
rotated_imgs = dtaug.rotate_images(dataset["images"], angles)
flipped_imgs = dtaug.flip_images(dataset["images"])

new_dataset["images"] = np.concatenate((new_dataset["images"], rotated_imgs))
new_dataset["images"] = np.concatenate((new_dataset["images"], flipped_imgs))

rotated_lbls = dtaug.rotate_images(dataset["labels"], angles)
flipped_lbls = dtaug.flip_images(dataset["labels"])

new_dataset["labels"] = np.concatenate((new_dataset["labels"], rotated_lbls))
new_dataset["labels"] = np.concatenate((new_dataset["labels"], flipped_lbls)).astype(dtype=np.int32)

new_dataset["classes"] = dataset["classes"]

print("Data Augmentation Applied:")
print("  -> Images: ", new_dataset["images"].shape)
print("  -> Labels: ", new_dataset["labels"].shape)


# # Split dataset between train, test and validation data

# In[ ]:


train_images, test_images, valid_images, train_labels, test_labels, valid_labels = dsutils.split_dataset(new_dataset)

print("Splitted dataset:")
print("  -> Train images: ", train_images.shape)
print("  -> Test images: ", test_images.shape)
print("  -> Validation images: ", valid_images.shape)
print("  -> Train Labels: ", train_labels.shape)
print("  -> Test Labels: ", test_labels.shape)
print("  -> Validation Labels: ", valid_labels.shape)


# # Train the Network

# In[ ]:


params = {
    "epochs": 600,
    "batch_size": 100,
    "learning_rate": 0.0001,
    "l2_reg_rate": 0.5,
    "var_scale_factor": 2.0,
    "chips_tensorboard": 2,
    "dropout_rate": 0.5,
    "class_names": dataset["classes"],
     "bands_plot": [1,2,3]
}


# In[ ]:


reload(mb)
model = mb.ModelBuilder(network)
model.train(train_images, test_images, train_labels, test_labels, params, model_dir)


# In[ ]:


#fcn.fcn_evaluate(valid_images, valid_labels,)

