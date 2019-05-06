import os
import sklearn


def compute_quality_metrics(labels, predictions, params):
    metrics = {'f1_score': sklearn.metrics.f1_score(labels, predictions, labels=[1, 2], average=None),
               'precision': sklearn.metrics.precision_score(labels, predictions, average=None),
               'recall': sklearn.metrics.recall_score(labels, predictions, average=None),
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

    out_str += 'Classification Report:' + os.linesep + str(metrics['classification_report']) + os.linesep
    out_str += 'Confusion Matrix:' + os.linesep + str(metrics['confusion_matrix']) + os.linesep

    return metrics, out_str
