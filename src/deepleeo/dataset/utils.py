from sklearn import model_selection
import numpy as np

def split_dataset(dataset, perc_test=30, perc_val=0, random_seed=None):
    train_images, test_images, train_labels, test_labels = model_selection.train_test_split(
        dataset["images"].astype(np.float32), #TODO: Verify if this casting is necessary
        dataset["labels"].astype(np.float32), #TODO: Verify if this casting is necessary
        test_size = ((perc_test + perc_val) / 100),
        random_state = random_seed
    )

    test_images, valid_images, test_labels, valid_labels = model_selection.train_test_split(
        test_images,
        test_labels,
        test_size = (perc_val / (perc_val + perc_test)),
        random_state = random_seed
    )

    return train_images, test_images, valid_images, train_labels, test_labels, valid_labels