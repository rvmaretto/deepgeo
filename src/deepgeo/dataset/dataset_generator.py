import numpy as np
import tensorflow as tf
import os
import sys
import scipy.misc
import pylab as pl
import sklearn
# from osgeo import gdal
# from osgeo import ogr
# from osgeo import osr  # TODO: Verify if it is really necessary? If I get the SRID from the Raster I still need this?

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))
import common.utils as utils
import common.filesystem as fs
import dataset.sequential_chips as seqchips
import dataset.random_chips as rdmchips
import dataset.fileset_chips as fset
import dataset.utils as dsutils


def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def wrap_float(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class DatasetGenerator(object):
    strategies = {
        'sequential': seqchips.SequentialChipGenerator,
        'random': rdmchips.RandomChipGenerator,
        'fileset': fset.FilesetChipGenerator
    }

    def __init__(self, raster_arrays, labels_arrays, strategy='sequential', description=None):
        if isinstance(raster_arrays, list):
            if len(raster_arrays) != len(labels_arrays):
                raise AttributeError('Lists "path_img" and "labeled_img" must have the same size!')
        else:
            raster_arrays = [raster_arrays]
            labels_arrays = [labels_arrays]

        for pos, lbl in enumerate(labels_arrays):
            if len(lbl.shape) < 3:
                labels_arrays[pos] = np.expand_dims(lbl.astype(np.int32), -1)

        self.raster_arrays = raster_arrays
        self.labels_arrays = labels_arrays
        self.strategy = strategy
        self.chips_struct = {}
        self.chip_size = 0
        self.description = description

    def generate_chips(self, params):
        print('  -> Generating chips...')
        self.chips_struct['chips'] = []
        self.chips_struct['labels'] = []
        self.chips_struct['coords'] = []
        for i in range(0, len(self.raster_arrays)):
            params['raster_array'] = self.raster_arrays[i]
            params['labels_array'] = self.labels_arrays[i]
            self.chip_size = params['win_size']

            chips_struct = self.strategies[self.strategy](params).generate_chips()

            self.chips_struct['chips'].append(chips_struct['chips'])
            self.chips_struct['labels'].append(chips_struct['labels'])
            if chips_struct['coords'] is not None:
                self.chips_struct['coords'] = self.chips_struct['coords'] + list(chips_struct['coords'])

        #WARNING: the below lines will lead to bug when chips_struct is a list with one element 
        self.chips_struct['chips'] = np.expand_dims(self.chips_struct['chips'], axis=0)        
        self.chips_struct['chips'] = np.concatenate(self.chips_struct['chips'], axis=0)
        
        self.chips_struct['labels'] = np.expand_dims(self.chips_struct['labels'], axis=0)
        self.chips_struct['labels'] = np.concatenate(self.chips_struct['labels'], axis=0)
        
        if 'overlap' in params:
            self.chips_struct['overlap'] = params['overlap']

    def get_samples(self):
        return self.chips_struct

    def remove_no_data(self, tolerance=.99):
        print('  -> Removing no data chips...')
        coords_remove = []
        for i in range(0, len(self.chips_struct['chips'])):
            if np.count_nonzero(self.chips_struct['labels'][i] == 0) > ((self.chip_size * self.chip_size) * tolerance):
                coords_remove.append(i)
        self.chips_struct['chips'] = np.delete(self.chips_struct['chips'], coords_remove, axis=0)
        self.chips_struct['labels'] = np.delete(self.chips_struct['labels'], coords_remove, axis=0)
        
        self.chips_struct['coords'] = [x for i, x in enumerate(self.chips_struct['coords']) if i not in coords_remove]

    def shuffle_ds(self):
        print('  -> Shuffling Dataset...')
        chips, labels = sklearn.utils.shuffle(self.chips_struct['chips'],
                                              self.chips_struct['labels'])
        self.chips_struct['chips'] = chips
        self.chips_struct['labels'] = labels

    def split_ds(self, perc_test=20, perc_val=20, random_seed=None):
        print('  -> Splitting Dataset...')
        train_chips, test_chips, val_chips, train_labels, test_labels, val_labels =\
            dsutils.split_dataset(self.chips_struct, perc_test, perc_val, random_seed)
        self.chips_struct = {'train': {'chips': train_chips, 'labels': train_labels},
                             'test': {'chips': test_chips, 'labels': test_labels},
                             'valid': {'chips': val_chips, 'labels': val_labels}}

    def save_to_disk(self, out_path, filename):
        print('  -> Saving Datasets to disk...')

        fs.mkdir(out_path)
        if self.description is not None:
            if 'train' in self.chips_struct:
                self.description['train_samples'] = self.chips_struct['train']['chips'].shape[0]
            if 'test' in self.chips_struct:
                self.description['test_samples'] = self.chips_struct['test']['chips'].shape[0]
            if 'valid' in self.chips_struct:
                self.description['valid_samples'] = self.chips_struct['valid']['chips'].shape[0]

            utils.save_dict_2_csv(self.description, os.path.join(out_path, 'description.csv'))

        if 'train' in self.chips_struct:
            suffixes = ['train', 'test']
        else:
            suffixes = ['']
        for suf in suffixes:
            chips = self.chips_struct[suf]
            out_file_path = os.path.join(out_path, filename + '_' + suf + '.tfrecord')
            with tf.io.TFRecordWriter(out_file_path) as writer:
                for pos in range(chips['chips'].shape[0]):
                    img = chips['chips'][pos, :, :, :]
                    lbl = chips['labels'][pos, :, :, :]

                    height = img.shape[0]
                    width = img.shape[1]
                    channels = img.shape[2]

                    img_raw = img.tostring()
                    lbl_raw = lbl.tostring()

                    feature = {'image': wrap_bytes(img_raw),
                               'label': wrap_bytes(lbl_raw),
                               'channels': wrap_int64(channels),
                               'height': wrap_int64(height),
                               'width': wrap_int64(width)}

                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

        out_file_path = os.path.join(out_path, filename + '_valid.npz')
        np.savez(out_file_path,
                 chips=self.chips_struct['valid']['chips'],
                 labels=self.chips_struct['valid']['labels'])
        print('  -> DONE!')

    def save_samples_PNG(self, path, color_map=None, r_g_b=[1, 2, 3]):
        for pos in range(len(self.samples_img)):
            samples_dir = os.path.join(path, 'sample_imgs')
            labels_dir = os.path.join(path, 'sample_labels')
            fs.mkdir(samples_dir)
            fs.mkdir(labels_dir)
            file_name = 'sample' + str(pos) + '.png'
            scipy.misc.imsave(os.path.join(samples_dir, file_name), self.samples_img[pos][:, :, r_g_b])
            if color_map is None:
                scipy.misc.imsave(os.path.join(labels_dir, file_name), self.samples_labels[pos][:, :, 0])
            else:
                pl.imsave(fname=os.path.join(labels_dir, file_name), arr=self.samples_labels[pos][:, :, 0],
                          cmap=color_map)

    def save_samples_NPZ(self, path, no_data=255):
        if os.path.exists(path):
            os.remove(path)
        np.savez(path,
                 images=self.samples_img,
                 labels=np.ma.filled(self.samples_labels, no_data),
                 classes=np.array(self.class_names))
