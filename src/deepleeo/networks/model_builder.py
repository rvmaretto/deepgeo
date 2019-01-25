import sys
import tensorflow as tf
import numpy as np
from os import path
import csv
from tensorflow.python.client import timeline

sys.path.insert(0, path.join(path.dirname(__file__),"../"))
import utils.filesystem as fs
import networks.fcn1s as fcn1s
import networks.fcn2s as fcn2s
import networks.fcn4s as fcn4s
import networks.fcn8s as fcn8s
import networks.fcn32s as fcn32s
import networks.unet as unet
import networks.laterfusion.unet_lf as unet_lf

#TODO: Remove this
def discretize_values(data, numberClass, startValue=0):
    for clazz in range(startValue, (numberClass + 1)):
        if clazz == startValue:
            classFilter = (data <= clazz + 0.5)
        elif clazz == numberClass:
            classFilter = (data > clazz - 0.5)
        else:
            classFilter = np.logical_and(data > clazz - 0.5, data <= clazz + 0.5)
        data[classFilter] = clazz

    return data.astype(np.uint8)

#TODO: Implement in the ModelBuilder a function that computes the output size.
class ModelBuilder(object):
    predefModels = {
        "fcn1s": fcn1s.fcn1s_description,
        "fcn2s": fcn2s.fcn2s_description,
        "fcn4s": fcn4s.fcn4s_description,
        "fcn8s": fcn8s.fcn8s_description,
        "fcn32s": fcn32s.fcn32s_description,
        "unet": unet.unet_description,
        "unet_lf": unet_lf.unet_lf_description
    }

    def __init__(self, model):
        if isinstance(model, str):
            self.network = model
            self.model_description = self.predefModels[model]
        else:
            self.network = "custom" #TODO: Change this. Implement a registration for new strategies.
            self.model_description = model

    def train(self, train_imgs, test_imgs, train_labels, test_labels, params, output_dir):
        # tf.set_random_seed(1987)
        tf.logging.set_verbosity(tf.logging.INFO)

        if not path.exists(output_dir):
            fs.mkdir(output_dir)
            
        with open(path.join(output_dir, "parameters.csv"), "w") as f:
            w = csv.writer(f, delimiter=';')
            w.writerow(["network", self.network])
            w.writerow(["input_chip_size", [train_imgs[0].shape[0], train_imgs[0].shape[1]]])
            w.writerow(["num_channels", train_imgs[0].shape[2]])
            for key, value in params.items():
                w.writerow([key, value])

        data_size, _, _, bands = train_imgs.shape
        params["bands"] = bands

        # print("UNIQUE LABELS: ", np.unique(train_labels))
        # print("UNIQUE IMAGE: ", np.unique(train_imgs))

        estimator = tf.estimator.Estimator(#model_fn=self.model_description,
                                        model_fn=tf.contrib.estimator.replicate_model_fn(self.model_description),
                                        model_dir=output_dir,
                                        params=params)

        tensors_to_log = {'loss': 'loss'}#,
                          # 'accuracy': 'accuracy'}#,
                          # 'learning_rate': 'learning_rate'}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10)

        print("Labels Shape: ", train_labels.shape)

        for epoch in range(1, params["epochs"] + 1):
            print("===============================================")
            print("Epoch ", epoch)
            train_input = tf.estimator.inputs.numpy_input_fn(x={"data": train_imgs},
                                                            y=train_labels,
                                                            batch_size=params["batch_size"],
                                                            num_epochs=1,
                                                            shuffle=True)

            print("---------------")
            print("Training...")
            train_results = estimator.train(input_fn=train_input, steps=None, hooks=[logging_hook])

            test_input = tf.estimator.inputs.numpy_input_fn(x={"data": test_imgs},
                                                            y=test_labels,
                                                            batch_size=params["batch_size"],
                                                            num_epochs=1,
                                                            shuffle=False)

            print("---------------")
            print("Evaluating...")
            test_results = estimator.evaluate(input_fn=test_input, hooks=[logging_hook])
        
        # tf.estimator.train_and_evaluate(estimator,
        #                                 train_spec=tf.estimator.TrainSpec(train_input, hooks=[logging_hook]),
        #                                 eval_spec=tf.estimator.EvalSpec(test_input))

    # def fcn_evaluate(images, labels, params, model_dir):
    #     data_size, _, _, _ = images.shape
    #
    #     tf.logging.set_verbosity(tf.logging.WARN)
    #
    #     estimator = tf.estimator.Estimator(model_fn=tf.contrib.estimator.replicate_model_fn(fcn32_VGG_description),
    #                                        model_dir=model_dir,
    #                                        params=params)
    #     logging_hook = tf.train.LoggingTensorHook(tensors={}, every_n_iter=data_size)
    #
    #     input_imgs

    def predict(self, images, params, model_dir):
        tf.logging.set_verbosity(tf.logging.WARN)

        if params["multi_gpu"]:
            estimator = tf.estimator.Estimator(model_fn=tf.contrib.estimator.replicate_model_fn(self.model_description),
                                            model_dir=model_dir,
                                            params=params)
        else:
            estimator = tf.estimator.Estimator(model_fn=self.model_description,
                                            model_dir=model_dir,
                                            params=params)

        if not isinstance(images, np.ndarray):
            images = np.stack(images).astype(np.float32)

        data_size, _, _ ,_ = images.shape
        input_fn = tf.estimator.inputs.numpy_input_fn(x={"data": images},
                                                    batch_size=params["batch_size"],
                                                    shuffle=False)

        predictions = estimator.predict(input_fn=input_fn)

        print("Classifying image with structure ", str(images.shape), "...")

        predicted_images = []

        for predict, dummy in zip(predictions, images):
            # predicted_images.append(np.argmax(predict["probabilities"], -1))
            # classif = np.argmax(predict["probabilities"], axis=-1)
            predicted_images.append(discretize_values(predict["classes"],
                                                      len(params["class_names"]),
                                                      0))

        return predicted_images
