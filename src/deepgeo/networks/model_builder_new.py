import os
import sys
import tensorflow.keras as keras

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
import common.utils as utils
import networks.dataset_loader as dsloader
import networks.unet_new as unet

class ModelBuilder(object):
    params = {
        'epochs': None,
        'batch_size': 10,
        'learning_rate': 0.001,
        'learning_rate_decay': True,
        'decay_rate': 0.1,
        'decay_steps': 245,
        'l2_reg_rate': 0.5,
        'dropout_rate': 0.5,
        'var_scale_factor': 2.0,
        # 'chips_tensorboard': 2,
        'fusion': 'none',
        'loss_func': 'categorical_crossentropy',
        'optimizer': 'Nadam',
        # 'bands_plot': [0, 1, 2]
        'metrics': ["accuracy", "categorical_accuracy", "AUC"] #, f1_m, recall_m, precision_m],#, "MeanIoU"]
    }

    predefModels = {
        # "fcn1s": fcn1s.fcn1s_description,
        # "fcn2s": fcn2s.fcn2s_description,
        # "fcn4s": fcn4s.fcn4s_description,
        # "fcn8s": fcn8s.fcn8s_description,
        # "fcn32s": fcn32s.fcn32s_description,
        "unet": unet.unet,
        # "unet_lf": unet_lf.unet_lf_description,
        # "mask_unet": mask_unet.mask_unet_description
    }

    loss_functions = {
        'categorical_crossentropy': 'categorical_crossentropy'
        # 'bin_iou': lossf.twoclass_cost,
        # 'avg_soft_dice': lossf.avg_soft_dice,
        # 'avg_generalized_dice': lossf.avg_generalized_dice,
        # 'weighted_cross_entropy': lossf.weighted_cross_entropy,
        # 'weighted_bin_cross_entropy': lossf.weighted_binary_cross_entropy
    }
    #
    # predefClassif = {
    #     'sigmoid': tf.nn.sigmoid,
    #     'softmax': tf.nn.softmax
    # }

    def __init__(self, params):
        if isinstance(params, dict):
            self.params = {**self.params, **params}
        elif isinstance(params, str):
            self.params = utils.read_csv_2_dict(os.path.join(params, 'parameters.csv'), keys_exclude=['dataset', 'Notes'])
        else:
            raise TypeError("Parameter 'params' must be a dictionary or a path to a .csv file")

        self.network = self.params['network']
        self.model_description = self.predefModels[self.params['network']]

        self.params['shape'] = (self.params["patch_size"], self.params["patch_size"], self.params['bands'])

        self.model = self.model_description(self.params)
        self.model.compile(loss=self.loss_functions[self.params['loss_func']],
                           metrics=self.params['metrics'],
                           optimizer=self.params['optimizer'])

    def register_loss(self, name, loss_func):
        self.loss_functions[name] = loss_func

    # Change it to the constructor.
    def register_model(self, name, model):
        self.params['network'] = name
        self.network = self.params['network']

        self.predefModels[self.params['network']] = model
        self.model_description = self.predefModels[self.params['network']]

    # def __build_model(self, params):
    #
    #     self.model = self.model_description(params)
    #     self.model.compile(loss=self.loss_functions[self.params['loss_func']],
    #                        metrics=params['metrics'],
    #                        optimizer=params['optimizer'])

    def train(self, train_dataset, test_dataset, output_dir):

        if self.params['epochs'] is None:
            raise ValueError("Field 'epochs' is not defined in 'params' dictionary.")

        train_loader = dsloader.DatasetLoader(train_dataset, self.params)
        test_loader = dsloader.DatasetLoader(test_dataset, self.params)
        number_of_chips = train_loader.get_dataset_size()
        self.params['number_of_chips'] = number_of_chips

        print('------------')
        print('Training with ', number_of_chips, ' chips...')

        # multpl_data_aug = 1
        # if 'data_aug_per_chip' in self.params:
        #     multpl_data_aug = self.params['data_aug_per_chip'] + 1
        # elif 'data_aug_ops' in self.params:
        #     multpl_data_aug = len(self.params['data_aug_ops']) + 1

        self.history = self.model.fit(train_loader.tfrecord_input_fn(),
                                 validation_data=test_loader.tfrecord_input_fn(train=False),
                                 batch_size=self.params['batch_size'],
                                 epochs=self.params['epochs'],
                                 verbose=1,
                                 callbacks=[keras.callbacks.TensorBoard(log_dir=output_dir, write_graph=True)]
                          # , write_images=True)]
                          # earlyStopCB = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                          #                                                min_delta=0,
                          #                                                patience=15,
                          #                                                verbose=1,
                          #                                                mode='auto',
                          #                                                baseline=None,
                          #                                                restore_best_weights=True)]
                          )

    # def predict(self, ):