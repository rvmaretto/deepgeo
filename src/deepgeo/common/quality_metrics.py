import gdal
import os
import sklearn
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
import common.visualization as vis


def compute_quality_metrics(labels, predictions, params, probabilities=None, classes_remove=[0]):
    class_names = params['class_names'].copy()
    labels = labels.flatten()
    predictions = predictions.flatten()
    if probabilities is not None:
        if len(probabilities.shape) < 4:
            probabilities = np.expand_dims(probabilities, axis=0)
        probabilities = probabilities[:, :, :, 2].flatten()
    else:
        probabilities = predictions

    for value in classes_remove:
        predictions = np.delete(predictions, np.where(labels == value))
        predictions[predictions == 0] = 1 # TODO: Find a beter way to solve this problem
        probabilities = np.delete(probabilities, np.where(labels == value))
        labels = np.delete(labels, np.where(labels==value))
        del class_names[value]
 
    metrics = {'f1_score': sklearn.metrics.f1_score(labels, predictions, labels=[1, 2], average=None),
               'precision': sklearn.metrics.precision_score(labels, predictions, average=None),
               'recall': sklearn.metrics.recall_score(labels, predictions, average=None),
               'roc_curve': sklearn.metrics.roc_curve(labels, probabilities, pos_label=2),  # TODO: try to put here one curve for each class
               'prec_rec_curve': sklearn.metrics.precision_recall_curve(labels, probabilities, pos_label=2),
               'classification_report': sklearn.metrics.classification_report(labels, predictions,
                                                                              target_names=class_names)}
    confusion_matrix = sklearn.metrics.confusion_matrix(labels, predictions, labels=[1, 2])
    metrics['confusion_matrix'] = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    metrics['auc_roc'] = sklearn.metrics.auc(metrics['roc_curve'][0], metrics['roc_curve'][1])
    #metrics['auc_prec_rec'] = sklearn.metrics.auc(metrics['prec_rec_curve'][0], metrics['prec_rec_curve'][1])

    out_str = ''
    out_str += 'F1-Score:' + os.linesep
    for i in range(0, len(metrics['f1_score'])):
        out_str += '  - ' + str(class_names[i]) + ': ' + str(metrics['f1_score'][i]) + os.linesep

    out_str += 'Precision:' + os.linesep
    for i in range(0, len(metrics['precision'])):
        out_str += '  - ' + str(class_names[i]) + ': ' + str(metrics['precision'][i]) + os.linesep

    out_str += 'Recall:' + os.linesep
    for i in range(0, len(metrics['recall'])): #TODO: check here when to use params and class_names
        out_str += '  - ' + str(class_names[i]) + ': ' + str(metrics['recall'][i]) + os.linesep

    # out_str += 'ROC: ' + os.linesep + '  - FPR: [ '
    # for i in metrics['roc_score'][0]:
    #     out_str += '{0}'.format(i) + ' '
    # out_str += ']' + os.linesep + '  - TPR: [ '
    # for i in metrics['roc_score'][1]:
    #     out_str += '{0}'.format(i) + ' '
    # out_str += ']' + os.linesep + '  - Thresholds: [ '
    # for i in metrics['roc_score'][2]:
    #     out_str += '{0}'.format(i) + ' '
    # out_str += ']' + os.linesep

    out_str += 'AUC-ROC: {0}'.format(metrics['auc_roc']) + os.linesep
    # out_str += 'AUC-Precision-Recall: {0}'.format(metrics['auc_prec_rec']) + os.linesep

    out_str += 'Classification Report:' + os.linesep + str(metrics['classification_report']) + os.linesep
    out_str += 'Confusion Matrix:' + os.linesep + str(metrics['confusion_matrix']) + os.linesep

    return metrics, out_str


def evaluate_classification(prediction_path, ground_truth_path, params, prediction_prob=None, out_dir=None, file_sufix=''):
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
        metrics, report_str = compute_quality_metrics(truth_arr, pred_arr, params, prob_arr)
    else:
        metrics, report_str = compute_quality_metrics(truth_arr, pred_arr, params)

    out_str += report_str

    print(out_str)

    if out_dir is not None:
        report_path = os.path.join(out_dir, ('classification_report' + file_sufix + '.txt'))
        out_file = open(report_path, 'w')
        out_file.write(out_str)
        out_file.close()

        conf_matrix_path = os.path.join(out_dir, ('classification_confusion_matrix' + file_sufix + '.png'))
        aucroc_curve_path = os.path.join(out_dir, ('auc_roc_curve' + file_sufix + '.png'))
        prec_rec_path = os.path.join(out_dir, ('prec_rec_curve' + file_sufix + '.png'))
    else:
        conf_matrix_path = None
        aucroc_curve_path = None
        prec_rec_path = None

    vis.plot_confusion_matrix(metrics['confusion_matrix'], params, conf_matrix_path)
    vis.plot_roc_curve(metrics['roc_curve'], aucroc_curve_path)
    vis.plot_precision_recall_curve(metrics['prec_rec_curve'], fig_path=prec_rec_path)

