import os
import sklearn
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
import common.visualization as vis
import dataset.utils as dsutils


def compute_quality_metrics(labels, predictions, params):
    labels = labels.flatten()
    predictions = predictions.flatten()
    metrics = {'f1_score': sklearn.metrics.f1_score(labels, predictions, labels=[1, 2], average=None),
               'precision': sklearn.metrics.precision_score(labels, predictions, average=None),
               'recall': sklearn.metrics.recall_score(labels, predictions, average=None),
               'roc_score': sklearn.metrics.roc_curve(labels, predictions, pos_label=2),
               'classification_report': sklearn.metrics.classification_report(labels, predictions,
                                                                              target_names=params['class_names'])}
    confusion_matrix = sklearn.metrics.confusion_matrix(labels, predictions, labels=[1, 2])
    metrics['confusion_matrix'] = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    out_str = ''
    out_str += 'F1-Score:' + os.linesep
    for i in range(0, len(metrics['f1_score'])):
        out_str += '  - ' + str(params['class_names'][i + 1]) + ': ' + str(metrics['f1_score'][i]) + os.linesep

    out_str += 'Precision:' + os.linesep
    for i in range(0, len(metrics['precision'])):
        out_str += '  - ' + str(params['class_names'][i]) + ': ' + str(metrics['precision'][i]) + os.linesep

    out_str += 'Recall:' + os.linesep
    for i in range(0, len(metrics['recall'])):
        out_str += '  - ' + str(params['class_names'][i]) + ': ' + str(metrics['recall'][i]) + os.linesep

    out_str += 'ROC: ' + os.linesep + '  - FPR: [ ' 
    for i in metrics['roc_score'][0]:
        out_str += '{0}'.format(i) + ' '  #metrics['roc_score'][0]) # + os.linesep
    out_str += ']' + os.linesep + '  - TPR: [ '
    for i in metrics['roc_score'][1]:
        out_str += '{0}'.format(i) + ' '
    out_str += ']' + os.linesep + '  - Thresholds: [ '
    for i in metrics['roc_score'][2]:
        out_str += '{0}'.format(i) + ' '
    out_str += ']' + os.linesep

    out_str += 'Classification Report:' + os.linesep + str(metrics['classification_report']) + os.linesep
    out_str += 'Confusion Matrix:' + os.linesep + str(metrics['confusion_matrix']) + os.linesep

    return metrics, out_str


def evaluate_classification(chip_struct, params, out_dir):
    num_chips, size_x, size_y, _ = chip_struct['predict'].shape
    crop_labels = dsutils.crop_np_batch(chip_struct['labels'], size_x)

    predictions = chip_struct['predict'].flatten()
    crop_labels = crop_labels.flatten()

    out_str = ''
    out_str += '<<------------------------------------------------------------>>' + os.linesep
    out_str += '<<---------------- Classification Results -------------------->>' + os.linesep
    out_str += '<<------------------------------------------------------------>>' + os.linesep

    metrics, report_str = compute_quality_metrics(crop_labels, predictions, params)

    out_str += report_str

    print(out_str)

    if out_dir is not None:
        report_path = os.path.join(out_dir, 'classification_report.txt')
        out_file = open(report_path, 'w')
        out_file.write(out_str)
        out_file.close()

        conf_matrix_path = os.path.join(out_dir, 'classification_confusion_matrix.png')

    vis.plot_confusion_matrix(metrics['confusion_matrix'], params, conf_matrix_path)
