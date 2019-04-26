import math
import sys
import numpy as np
from os import path
import tensorflow as tf
import csv

sys.path.insert(0, path.join(path.dirname(__file__), '../'))
import common.filesystem as fs
import networks.fcn1s as fcn1s
import networks.fcn2s as fcn2s
import networks.fcn4s as fcn4s
import networks.fcn8s as fcn8s
import networks.fcn32s as fcn32s
import networks.unet as unet
import networks.laterfusion.unet_lf as unet_lf
import networks.loss_functions as lossf
import networks.tb_metrics as tbm
import networks.layers as layers


# TODO: Remove this
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

# TODO: Implement in the ModelBuilder a function that computes the output size.
class ModelBuilder(object):
    default_params = {
        'epochs': None,
        'batch_size': 10,
        'learning_rate': 0.001,
        'learning_rate_decay': True,
        'decay_rate': 0.1,
        'decay_steps': 245,
        'l2_reg_rate': 0.5,
        'dropout_rate': 0.5,
        'var_scale_factor': 2.0,
        'chips_tensorboard': 2,
        'fusion': 'none',
        'loss_func': 'crossentropy',
        'bands_plot': [0, 1, 2]
    }

    predefModels = {
        "fcn1s": fcn1s.fcn1s_description,
        "fcn2s": fcn2s.fcn2s_description,
        "fcn4s": fcn4s.fcn4s_description,
        "fcn8s": fcn8s.fcn8s_description,
        "fcn32s": fcn32s.fcn32s_description,
        "unet": unet.unet_description,
        "unet_lf": unet_lf.unet_lf_description
    }

    losses_switcher = {
        'cross_entropy': tf.losses.softmax_cross_entropy,
        'weighted_crossentropy': lossf.weighted_cross_entropy,
        'soft_dice': lossf.avg_soft_dice
    }

    predefClassif = {
        'sigmoid': tf.nn.sigmoid,
        'softmax': tf.nn.softmax
    }

    def __init__(self, model):
        if isinstance(model, str):
            self.network = model
            self.model_description = self.predefModels[model]
        else:
            self.network = "custom"  # TODO: Change this. Implement a registration for new strategies.
            self.model_description = model

    def __build_model(self, features, labels, params, mode, config):
        tf.logging.set_verbosity(tf.logging.INFO)
        training = mode == tf.estimator.ModeKeys.TRAIN
        # global_step = tf.Variable(0, name='global_step', trainable=False)
        samples = features['data']

        logits = self.model_description(samples, labels, params, mode, config)

        if labels.shape[1] != logits.shape[1]:
            labels = tf.cast(layers.crop_features(labels, logits.shape[1], name="labels"), tf.float32)

        predictions = tf.nn.softmax(logits, name='Softmax')
        output = tf.expand_dims(tf.argmax(input=predictions, axis=-1, name='Argmax_Prediction'), -1)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=output)

        labels_1hot = tf.one_hot(tf.cast(labels, tf.uint8), params['num_classes'])
        labels_1hot = tf.squeeze(labels_1hot)

        # loss_params = {
        #     'logits': logits,
        #     'predictions': predictions,
        #     'output': output,
        #     'labels_1hot': labels_1hot,
        #     'labels': labels,
        #     'class_weights': params['class_weights'],
        #     'num_classes': params['num_classes']
        # }

        # loss = tf.losses.sigmoid_cross_entropy(labels_1hot, output)
        # loss = lossf.weighted_binary_cross_entropy(logits, labels, params['class_weights'])
        # loss = tf.losses.softmax_cross_entropy(labels_1hot, logits)
        # loss = lossf.twoclass_cost(output, labels)
        # loss = lossf.inverse_mean_iou(logits, labels_1hot, num_classes)
        # loss = lossf.avg_soft_dice(logits, labels_1hot)
        loss = lossf.weighted_cross_entropy(logits, labels_1hot, params['class_weights'], params['num_classes'],
                                            training)
        # loss_func = self.losses_switcher.get(params['loss_func'], lossf.unknown_loss_error)
        # loss = loss_func(loss_params)

        tbm.plot_chips_tensorboard(samples, labels, tf.expand_dims(predictions[:, :, :, 2], -1), params)
        metrics, summaries = tbm.define_quality_metrics(labels_1hot, predictions, logits, labels, output, loss, params)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        if params['learning_rate_decay']:
            params['learning_rate'] = tf.train.exponential_decay(learning_rate=params['learning_rate'],
                                                                 global_step=tf.train.get_global_step(),
                                                                 decay_rate=params['decay_rate'],
                                                                 decay_steps=params['decay_steps'],
                                                                 name='decrease_lr')

        tf.summary.scalar('learning_rate', params['learning_rate'])

        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'], name='Optimizer')
        # optimizer = tf.contrib.opt.NadamOptimizer(params['learning_rate'], name='Optimizer')
        # optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        if training:
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        else:
            train_op = None

        train_summary_hook = tf.train.SummarySaverHook(save_steps=100,
                                                       output_dir=config.model_dir,
                                                       summary_op=tf.summary.merge_all())

        eval_metric_ops = {'eval_metrics/accuracy': metrics['accuracy'],
                           'eval_metrics/f1-score': metrics['f1_score'],
                           'eval_metrics/cross_entropy': metrics['cross_entropy']}  # ,
                           # 'eval_metrics/mean_iou': metrics['mean_iou']}

        logging_hook = tf.train.LoggingTensorHook({'loss': loss,
                                                   'accuracy': metrics['accuracy'][1],
                                                   'f1_score': metrics['f1_score'][1],
                                                   'cross_entropy': metrics['cross_entropy'][1],
                                                   # 'mean_iou': metrics['mean_iou'][0],
                                                   'learning_rate': params['learning_rate']},
                                                  every_n_iter=100)

        eval_summary_hook = tf.train.SummarySaverHook(save_steps=100,
                                                      output_dir=config.model_dir + "/eval",
                                                      summary_op=tf.summary.merge_all())

        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=output,
                                          loss=loss,
                                          train_op=train_op,
                                          eval_metric_ops=eval_metric_ops,
                                          evaluation_hooks=[eval_summary_hook, logging_hook],
                                          training_hooks=[train_summary_hook, logging_hook])

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
        params['bands'] = bands
        params['decay_steps'] = math.ceil(data_size / params['batch_size'])

        # Try to update save the dataset as TFRecords and try to use this:
        # https://www.tensorflow.org/guide/distribute_strategy
        # strategy = tf.contrib.distribute.MirroredStrategy()
        # config = tf.estimator.RunConfig(train_distribute=strategy)#, eval_distribute=strategy)

        # TODO: Verify why it is breaking here
        # with tf.contrib.tfprof.ProfileContext(path.join(output_dir, "profile")) as pctx:
        estimator = tf.estimator.Estimator(model_fn=self.__build_model,
                                        #model_fn=tf.contrib.estimator.replicate_model_fn(self.__build_model),
                                        model_dir=output_dir,
                                        params=params)#,
                                        # config=config)

        #profiling_hook = tf.train.ProfilerHook(save_steps=10, output_dir=path.join(output_dir))


        # train_input = tf.data.Dataset.from_tensor_slices(({"x": train_imgs}, train_labels)).shuffle(buffer_size=2048)
        # train_input = train_input.shuffle(1000).repeat().batch(params["batch_size"])
        #
        # test_input = tf.data.Dataset.from_tensor_slices(({"x": test_imgs}, test_labels)).shuffle(buffer_size=2048)
        # test_input = test_input.shuffle(1000).repeat().batch(params["batch_size"])

        for epoch in range(1, params["epochs"] + 1):
            print("===============================================")
            print("Epoch ", epoch)
            train_input = tf.estimator.inputs.numpy_input_fn(x={"data": train_imgs},
                                                             y=train_labels,
                                                             batch_size=params["batch_size"],
                                                             num_epochs=1,  # params["epochs"],
                                                             shuffle=True)
            # train_input, train_init_hook = ds_it.get_input_fn(train_imgs, train_labels, params["batch_size"], shuffle=True)

            print("---------------")
            print("Training...")
            train_results = estimator.train(input_fn=train_input,
                                            steps=None)
                                            # hooks=[profiling_hook])

            test_input = tf.estimator.inputs.numpy_input_fn(x={"data": test_imgs},
                                                            y=test_labels,
                                                            batch_size=params["batch_size"],
                                                            num_epochs=1,#params["epochs"],
                                                            shuffle=False)
            
            # test_input, test_init_hook = ds_it.get_input_fn(test_imgs, test_labels, params["batch_size"], shuffle=True)

            print("---------------")
            print("Evaluating...")
            test_results = estimator.evaluate(input_fn=test_input)#,
                                              # hooks=[logging_hook])#, profiling_hook])

        # early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
        #     estimator,
        #     metric_name='cost/loss',
        #     max_steps_without_decrease=1000,
        #     eval_dir=path.join(output_dir, "eval"),
        #     min_steps=100)

        # tf.estimator.train_and_evaluate(estimator,
        #                                 train_spec=tf.estimator.TrainSpec(train_input, hooks=[logging_hook]),
        #                                 eval_spec=tf.estimator.EvalSpec(test_input, hooks=[logging_hook]))

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
            # predicted_images.append(discretize_values(predict["classes"],
            #                                           params["num_classes"],
            #                                           0))
            predicted_images.append(predict)

        return predicted_images
