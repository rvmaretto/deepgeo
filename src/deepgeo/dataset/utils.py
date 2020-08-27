from sklearn import model_selection
import numpy as np


def split_dataset(dataset, perc_test=30, perc_val=0, random_seed=None):
    train_images, test_images, train_labels, test_labels = model_selection.train_test_split(
        dataset['chips'],
        dataset['labels'],
        test_size=((perc_test + perc_val) / 100),
        random_state=random_seed
    )

    test_images, valid_images, test_labels, valid_labels = model_selection.train_test_split(
        test_images,
        test_labels,
        test_size = (perc_val / (perc_val + perc_test)),
        random_state = random_seed
    )

    return train_images, test_images, valid_images, train_labels, test_labels, valid_labels


def crop_np_chip(chip, out_size):
    feat_shape = chip.shape
    offsets = [int((int(feat_shape[0]) - int(out_size)) / 2),
               int((int(feat_shape[1]) - int(out_size)) / 2)]
    # size = [out_size, out_size, feat_shape[2]]
    # chip = np.slice(chip, offsets, size)
    chip = chip[offsets[0]:(offsets[0] + out_size), offsets[1]:(offsets[1] + out_size), :]
    return chip


def crop_np_batch(batch, out_size):
    feat_shape = batch.shape
    offsets = [int((int(feat_shape[1]) - int(out_size)) / 2),
               int((int(feat_shape[2]) - int(out_size)) / 2)]
    batch = batch[:, offsets[0]:(offsets[0] + out_size), offsets[1]:(offsets[1] + out_size), :]
    return batch
