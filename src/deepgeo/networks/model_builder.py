import csv
import math
import numpy as np
import tensorflow as tf
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
import common.filesystem as fs
import common.quality_metrics as qm
import common.visualization as vis
import dataset.utils as dsutils
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
import networks.dataset_loader as dsloader


# TODO: Remove this
def discretize_values(data, number_class, start_value=0):
    for clazz in range(start_value, (number_class + 1)):
        if clazz == start_value:
            class_filter = (data <= clazz + 0.5)
        elif clazz == number_class:
            class_filter = (data > clazz - 0.5)
        else:
            class_filter = np.logical_and(data > clazz - 0.5, data <= clazz + 0.5)
        data[class_filter] = clazz

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
        samples = features

        logits = self.model_description(samples, labels, params, mode, config)

        predictions = tf.nn.softmax(logits, name='Softmax')
        output = tf.expand_dims(tf.argmax(input=predictions, axis=-1, name='Argmax_Prediction'), -1)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions={'classes': output,
                                                                      'probabilities': predictions})

        if labels.shape[1] != logits.shape[1]:
            labels = tf.cast(layers.crop_features(labels, logits.shape[1], name="labels"), tf.float32)

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
                           'eval_metrics/cross_entropy': metrics['cross_entropy'],
                           'eval_metrics/auc_roc': metrics['auc-roc']}  # ,
                           # 'eval_metrics/mean_iou': metrics['mean_iou']}

        logging_hook = tf.train.LoggingTensorHook({'loss': loss,
                                                   'accuracy': metrics['accuracy'][1],
                                                   'f1_score': metrics['f1_score'][1],
                                                   'cross_entropy': metrics['cross_entropy'][1],
                                                   # 'mean_iou': metrics['mean_iou'][0],
                                                   'learning_rate': params['learning_rate'],
                                                   'auc_roc': metrics['auc-roc'][1]},
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

    def train(self, train_dataset, test_dataset, params, output_dir):
        # tf.set_random_seed(1987)
        tf.logging.set_verbosity(tf.logging.INFO)

        if not os.path.exists(output_dir):
            fs.mkdir(output_dir)
            
        with open(os.path.join(output_dir, "parameters.csv"), "w") as f:
            w = csv.writer(f, delimiter=';')
            w.writerow(["network", self.network])
            for key, value in params.items():
                w.writerow([key, value])

        params['shape'] = [params['chip_size'], params['chip_size'], params['bands']]
        train_loader = dsloader.DatasetLoader(train_dataset, params)
        test_loader = dsloader.DatasetLoader(test_dataset, params)
        number_of_chips = train_loader.get_dataset_size()
        params['number_of_chips'] = number_of_chips

        params['decay_steps'] = math.ceil((number_of_chips * len(params['data_aug_ops'])) / params['batch_size'])

        # https://www.tensorflow.org/guide/distribute_strategy
        strategy = tf.contrib.distribute.MirroredStrategy()
        config = tf.estimator.RunConfig(train_distribute=strategy)  # , eval_distribute=strategy)

        estimator = tf.estimator.Estimator(model_fn=self.__build_model,
                                           model_dir=output_dir,
                                           params=params,
                                           config=config)

        trainer = tf.estimator.TrainSpec(lambda: train_loader.tfrecord_input_fn())
        evaluator = tf.estimator.EvalSpec(lambda: test_loader.tfrecord_input_fn(train=False))
        tf.estimator.train_and_evaluate(estimator, train_spec=trainer, eval_spec=evaluator)

        # profiling_hook = tf.train.ProfilerHook(save_steps=10, output_dir=path.join(output_dir))

        # early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
        #     estimator,
        #     metric_name='cost/loss',
        #     max_steps_without_decrease=1000,
        #     eval_dir=path.join(output_dir, "eval"),
        #     min_steps=100)

    def validate(self, images, expect_labels, params, model_dir, save_results=True,
                 show_plots=True, exclude_classes=None):
        tf.logging.set_verbosity(tf.logging.WARN)

        out_dir = os.path.join(model_dir, 'validation')

        estimator = tf.estimator.Estimator(model_fn=self.__build_model,
                                           model_dir=model_dir,
                                           params=params)

        input_fn = tf.estimator.inputs.numpy_input_fn(x=images,
                                                      batch_size=params['batch_size'],
                                                      shuffle=False)

        predictions_lst = []
        probabilities_lst = []
        crop_labels = []
        for predict, label in zip(estimator.predict(input_fn), expect_labels):
            predictions_lst.append(predict['classes'])
            probabilities_lst.append(predict['probabilities'])
            size_x, size_y, _ = predict['classes'].shape
            label = dsutils.crop_np_chip(label, size_x)
            crop_labels.append(label)

        predictions = np.array(predictions_lst, dtype=np.int32)
        crop_labels = np.array(crop_labels, dtype=np.int32)
        probabilities = np.array(probabilities_lst, dtype=np.float32)

        out_str = ''
        out_str += '<<------------------------------------------------------------>>' + os.linesep
        out_str += '<<------------------ Validation Results ---------------------->>' + os.linesep
        out_str += '<<------------------------------------------------------------>>' + os.linesep

        metrics, report_str = qm.compute_quality_metrics(crop_labels, predictions, params, probabilities)

        out_str += report_str
        print(out_str)

        if save_results:
            fs.mkdir(os.path.join(model_dir, 'validation'))
            report_path = os.path.join(out_dir, 'validation_report.txt')
            out_file = open(report_path, 'w')
            out_file.write(out_str)
            out_file.close()

            conf_matrix_path = os.path.join(out_dir, 'validation_confusion_matrix.png')
            auc_roc_path = os.path.join(out_dir, 'auc_roc_curve.png')
            prec_rec_path = os.path.join(out_dir, 'prec_rec_curve.png')
        else:
            conf_matrix_path = None
            auc_roc_path = None
            prec_rec_path = None

        vis.plot_confusion_matrix(metrics['confusion_matrix'], params, conf_matrix_path, show_plot=show_plots)
        vis.plot_roc_curve(metrics['roc_curve'], auc_roc_path, show_plot=show_plots)
        vis.plot_precision_recall_curve(metrics['prec_rec_curve'], fig_path=prec_rec_path, show_plot=show_plots)

    def predict(self, chip_struct, params, model_dir, return_prob=True):
        tf.logging.set_verbosity(tf.logging.WARN)
        images = chip_struct['chips']

        estimator = tf.estimator.Estimator(model_fn=self.__build_model,
                                           model_dir=model_dir,
                                           params=params)

        input_fn = tf.estimator.inputs.numpy_input_fn(x=images,
                                                      batch_size=params['batch_size'],
                                                      shuffle=False)

        print('Classifying image with structure ', str(images.shape), '...')

        predictions = []
        if return_prob:
            probabilities = []

        for predict in estimator.predict(input_fn):
            predictions.append(predict['classes'])
            if return_prob:
                probabilities.append(predict['probabilities'])
        chip_struct['predict'] = np.array(predictions, dtype=np.int32)
        if return_prob:
            chip_struct['probabilities'] = np.array(probabilities, dtype=np.float32)

        return chip_struct

