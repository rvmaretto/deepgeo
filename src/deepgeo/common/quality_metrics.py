from osgeo import gdal
import os
import sklearn
import sys
import numpy as np
import sklearn.metrics as metrics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
import common.visualization as vis


def compute_quality_metrics(labels, predictions, params, probabilities=None, classes_ignore=[0]):
    class_names = params['class_names'].copy()
    labels = labels.flatten()
    predictions = predictions.flatten()
    if probabilities is not None:
        if len(probabilities.shape) < 4:
            probabilities = np.expand_dims(probabilities, axis=0)
        probabilities = probabilities.reshape(-1, probabilities.shape[-1])
    # else:
    #     probabilities = predictions

    for value in classes_ignore:
        predictions = np.delete(predictions, np.where(labels == value))
        print('UNIQUE Predictions: ', np.unique(predictions))
        predictions[predictions == 0] = 1  # TODO: Find a better way to solve this problem
        if probabilities is not None:
            probabilities = np.delete(probabilities, np.where(labels == value), axis=0)
        labels = np.delete(labels, np.where(labels == value))
        del class_names[value]

    labels_to_use = []
    for clazz in class_names:
        labels_to_use.append(params['class_names'].index(clazz))
 
    metrics_dict = {}
    with sklearn.utils.parallel_backend('multiprocessing'):
    # with sklearn.externals.joblib.parallel_backend('multiprocessing'):
        metrics_dict['f1_score'] = metrics.f1_score(labels, predictions, labels=labels_to_use, average=None)
        metrics_dict['precision'] = metrics.precision_score(labels, predictions, average=None)
        metrics_dict['recall'] = metrics.recall_score(labels, predictions, average=None)
        metrics_dict['accuracy'] = metrics.accuracy_score(labels, predictions)
        metrics_dict['classification_report'] = metrics.classification_report(labels, predictions,
                                                                                 target_names=class_names,
                                                                                 digits=4)

        if probabilities is not None:
            metrics_dict['prec_rec_curve'] = {}
            for clazz in class_names:
                cl_index = params['class_names'].index(clazz)
                metrics_dict['prec_rec_curve'][clazz] = metrics.precision_recall_curve(labels,
                                                                                       probabilities[:, cl_index],
                                                                                       pos_label=cl_index)

            metrics_dict['roc_curve'] = {}
            metrics_dict['auc_roc'] = {}
            for clazz in class_names:
                cl_index = params['class_names'].index(clazz)
                metrics_dict['roc_curve'][clazz] = metrics.roc_curve(labels,
                                                                     probabilities[:, cl_index],
                                                                     pos_label=cl_index)
                metrics_dict['auc_roc'][clazz] = metrics.auc(metrics_dict['roc_curve'][clazz][0], metrics_dict['roc_curve'][clazz][1])

        confusion_matrix = metrics.confusion_matrix(labels, predictions, labels=labels_to_use)
        metrics_dict['confusion_matrix'] = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    out_str = ''
    out_str += 'F1-Score:' + os.linesep
    for i in range(0, len(metrics_dict['f1_score'])):
        out_str += '  - ' + str(class_names[i]) + ': ' + str(metrics_dict['f1_score'][i]) + os.linesep

    out_str += 'Precision:' + os.linesep
    for i in range(0, len(metrics_dict['precision'])):
        out_str += '  - ' + str(class_names[i]) + ': ' + str(metrics_dict['precision'][i]) + os.linesep

    out_str += 'Recall:' + os.linesep
    for i in range(0, len(metrics_dict['recall'])):
        out_str += '  - ' + str(class_names[i]) + ': ' + str(metrics_dict['recall'][i]) + os.linesep

    out_str += 'Accuracy: ' + str(metrics_dict['accuracy']) + os.linesep

    if probabilities is not None:
        for clazz, val in metrics_dict['auc_roc'].items():
            out_str += 'AUC-ROC {}: {}'.format(clazz, metrics_dict['auc_roc']) + os.linesep

    out_str += 'Classification Report:' + os.linesep + str(metrics_dict['classification_report']) + os.linesep
    out_str += 'Confusion Matrix:' + os.linesep + str(metrics_dict['confusion_matrix']) + os.linesep

    return metrics_dict, out_str


def evaluate_classification(prediction_path, ground_truth_path, params, prediction_prob=None,
                            out_dir=None, file_sufix='', classes_ignore=[0]):
    pred_ds = gdal.Open(prediction_path)
    pred_arr = pred_ds.ReadAsArray()
    pred_size_x, pred_size_y = pred_arr.shape

    if prediction_prob is not None:
        prob_ds = gdal.Open(prediction_prob)
        prob_arr = prob_ds.ReadAsArray()
        prob_arr = np.rollaxis(prob_arr, 0, 3)

    truth_ds = gdal.Open(ground_truth_path)
    truth_arr = truth_ds.ReadAsArray()
    truth_size_x, truth_size_y = truth_arr.shape

    if pred_size_x != truth_size_x:
        diff_x = truth_size_x - pred_size_x
        diff_y = truth_size_y - pred_size_y
        start_x = 0 + round(diff_x / 2)
        end_x = truth_size_x - round(diff_x / 2)
        start_y = 0 + round(diff_y / 2)
        end_y = truth_size_y - round(diff_y / 2)
        truth_arr = truth_arr[start_x:end_x, start_y:end_y]

    out_str = ''
    out_str += '<<------------------------------------------------------------>>' + os.linesep
    out_str += '<<---------------- Classification Results -------------------->>' + os.linesep
    out_str += '<<------------------------------------------------------------>>' + os.linesep

    if prediction_prob is not None:
        metrics_dict, report_str = compute_quality_metrics(truth_arr, pred_arr, params, prob_arr,
                                                      classes_ignore=classes_ignore)
    else:
        metrics_dict, report_str = compute_quality_metrics(truth_arr, pred_arr, params, classes_ignore=classes_ignore)

    out_str += report_str

    print(out_str)

    if out_dir is not None:
        report_path = os.path.join(out_dir, ('classification_report' + file_sufix + '.txt'))
        out_file = open(report_path, 'w')
        out_file.write(out_str)
        out_file.close()

        conf_matrix_path = os.path.join(out_dir, ('classification_confusion_matrix' + file_sufix + '.png'))
        if prediction_prob is not None:
            aucroc_curve_path = os.path.join(out_dir, ('auc_roc_curve' + file_sufix + '.png'))
            prec_rec_path = os.path.join(out_dir, ('prec_rec_curve' + file_sufix + '.png'))
    else:
        conf_matrix_path = None
        aucroc_curve_path = None
        prec_rec_path = None

    vis.plot_confusion_matrix(metrics_dict['confusion_matrix'], params, classes_remove=classes_ignore,
                              fig_path=conf_matrix_path)
    if prediction_prob is not None:
        vis.plot_roc_curve(metrics_dict['roc_curve'], aucroc_curve_path)
        vis.plot_precision_recall_curve(metrics_dict['prec_rec_curve'], fig_path=prec_rec_path)

