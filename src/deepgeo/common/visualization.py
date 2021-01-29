import numpy
import numpy as np
import os
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import skimage
import sklearn
from descartes import PolygonPatch
from osgeo import ogr
from shapely.wkb import loads
from skimage import exposure
import palettable
# from shapely.geometry import Polygon


def plot_rgb_img(raster_array, bands=[1, 2, 3], contrast=False, title="RGB Composition", fig_path=None,
                 figsize=(10, 10), onlyreturn=False):
    if len(bands) != 3 and len(bands) != 1:
        raise AttributeError("Parameter bands must have size 3 or 1.")
    
    raster_img = skimage.img_as_ubyte(raster_array)
    if contrast:
        for band in bands:
            #TODO: How to keep the array masked here?
            p2, p98 = np.percentile(raster_img[:, :, band], (2, 98))
            raster_img[:, :, band] = exposure.rescale_intensity(raster_img[:, :, band], in_range=(p2, p98))
    if onlyreturn:
        return raster_img[:, :, bands]
    else:    
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.axis('off')
        if len(bands) == 3:
            plt.imshow(raster_img[:, :, bands])
        else:
            plt.imshow(raster_img[:,:,bands[0]])
        if fig_path is not None:
            plt.savefig(fig_path)
        plt.show()


def plot_labels(labels_array, class_names, colors=None, title="Labels", fig_path=None, figsize=(10, 10)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plt.title(title)

    # labels = np.ma.masked_where(labels_array == 255, labels_array) # TODO: Is this line necessary? Try to comment it

    num_classes = len(class_names)
    bounds = [-0.1] + list(range(0, num_classes)) + [num_classes - 0.9]
    if colors is None:
        colorMap = plt.cm.get_cmap("prism", num_classes)
    else:
        colorMap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, colorMap.N) 

    if len(labels_array.shape) > 2:
        ax.imshow(labels_array[:,:,0], cmap=colorMap, norm=norm)
    else:
        ax.imshow(labels_array, cmap=colorMap, norm=norm)
    # cbar = plt.colorbar()  # OLD legend
    # cbar.ax.get_yaxis().set_ticks([])  # OLD legend
    plt.axis('off')

    leg_handles = []
    for j, lab in enumerate(class_names):
        # cbar.ax.text(1.5, (2 * j + 1) / (num_classes * 2), lab, ha='left')  # OLD legend
        leg_handles.append(patches.Patch(color=colors[j], label=lab))

    leg = ax.legend(handles=leg_handles, loc='best', bbox_to_anchor=(1.22, 0.15), fontsize='x-large')
    # cbar.ax.get_yaxis().labelpad = 15  # OLD legend
    fig.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path, bbox_extra_artists=(leg,), bbox_inches='tight')


# TODO: How to plot the raster together? Decrease the blank space from the origin to the data
def plot_vector_file(path_file, edge_color="black", face_color="red"):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    out_ds = ogr.Open(path_file)

    layer = out_ds.GetLayerByName(os.path.splitext(os.path.basename(path_file))[0])

    min_x, max_x, min_y, max_y = layer.GetExtent()
    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])

    parcel = layer.GetNextFeature()

    while parcel is not None:
        polygon = loads(parcel.GetGeometryRef().ExportToWkb())
        # xCoord, yCoord = polygon.exterior.xy
        # ax.fill(xCoord, yCoord, "r")
        # ax.plot(xCoord, yCoord, "k-")
        ring_patch = PolygonPatch(polygon, edgecolor=edge_color, facecolor=face_color)
        ax.add_patch(ring_patch)
        parcel = layer.GetNextFeature()

    out_ds.Destroy()


def plot_image_histogram(raster_array, cmap=None, nbins = 256, title="Histogram", legend=None):
    fig = plt.figure(figsize=(12, 8))
    plt.title(title, fontsize=15)
    #plt.ylim([0,100000])
    plt.xlabel("Bins", fontsize=12)
    plt.ylabel("Number of Pixels", fontsize=12)

    if len(raster_array.shape) > 2:
        num_channels = raster_array.shape[2]
    else:
        num_channels = 1

    if cmap is None:
        cmap = plt.cm.get_cmap("hsv", num_channels)
    elif isinstance(cmap, list):
        cmap = mcolors.ListedColormap(cmap)

    if legend is None:
        legend = []
        for i in range(num_channels):
            legend.append("Channel " + str(i))

    if len(raster_array.shape) > 2:
        for band in range(num_channels):
            plt.hist(raster_array[:,:,band].ravel(), nbins, color=cmap(band), alpha=0.4)

        plt.legend(legend)
    else:
        plt.hist(raster_array.ravel(), color=cmap(0), alpha=0.4)

    plt.show()


def plot_image_histogram_lines(raster_array, cmap=None, title="Histogram", legend=None):
    fig = plt.figure(figsize=(12, 8))
    plt.title(title, fontsize=15)
    plt.xlabel("Bins", fontsize=12)
    plt.ylabel("Number of Pixels", fontsize=12)

    if len(raster_array.shape) > 2:
        num_channels = raster_array.shape[2]
    else:
        num_channels = 1

    if (cmap is None):
        cmap = plt.cm.get_cmap("hsv", num_channels)
    elif isinstance(cmap, list):
        cmap = mcolors.ListedColormap(cmap)

    if legend is None:
        legend = []
        for i in range(num_channels):
            legend.append("Channel " + str(i))

    if len(raster_array.shape) > 2:
        for band in range(num_channels):
            sns.kdeplot(raster_array[:, :, band].ravel(), color=cmap(band), label=legend[band])
    else:
        sns.kdeplot(raster_array.ravel(), color=cmap(0), label=legend[0])

    plt.show()


def plot_chips(chips, raster_array=None, bands=[1, 2, 3], contrast=False, chipscolor='blue'):
    fig, ax = plt.subplots(1, figsize=(12, 12))

    # Display the image
    if raster_array is not None:
        # raster_img = skimage.img_as_ubyte(raster_array)
        raster_img = raster_array
        if contrast:
            for band in bands:
                p2, p98 = np.percentile(raster_img[:, :, band], (2, 98))
                raster_img[:, :, band] = exposure.rescale_intensity(raster_img[:, :, band], in_range=(p2, p98))

        if len(bands) == 3:
            ax.imshow(raster_img[:, :, bands])
        else:
            ax.imshow(raster_img[:, :, bands[0]])

        plt.axis('off')

    for coord in chips['coords']:
        width = coord['lower_row'] - coord['upper_row']
        height = coord['right_col'] - coord['left_col']
        rect = patches.Rectangle((coord['left_col'], coord['upper_row']), width, height,
                                  edgecolor=chipscolor, facecolor='none')
        ax.add_patch(rect)

    # plt.savefig('/home/raian/doutorado/PA_chips_small.pdf')  #TODO: Include this in the parameters of the function
    # plt.show()


def plot_confusion_matrix(confusion_matrix, params, classes_remove=[0], fig_path=None, show_plot=True):
    class_names = params['class_names'].copy()
    for value in classes_remove:
        del class_names[value]
    fig, ax = plt.subplots(figsize=(12, 12))
    img = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Greens)
    ax.figure.colorbar(img, ax=ax)
    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.4f'
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(confusion_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    fig.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path)
    if show_plot:
        plt.show()
    else:
        plt.clf()


def plot_roc_curve(roc, fig_path=None, show_plot=True):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for clazz, roc_values in roc.items():
        fpr = roc_values[0]
        tpr = roc_values[1]
        thresholds = roc_values[2]

        tprs.append(scipy.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

        roc_auc = sklearn.metrics.auc(fpr, tpr)
        aucs.append(roc_auc)

        ax.plot(fpr, tpr, label='ROC %s (AUC = %0.4f)' % (clazz, roc_auc), lw=2, alpha=.8)

    ax.plot([0, 1], [0, 1], linestyle='--', color='r', lw=2, label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, label='Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristics')
    legend = ax.legend(loc='best', bbox_to_anchor=(1.05, 1))
    if fig_path is not None:
        plt.savefig(fig_path, bbox_extra_artists=(legend,), bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.clf()


def plot_precision_recall_curve(prec_rec, fig_path=None, show_plot=True):
    # precision = {}
    # recall = {}
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for clazz, pr_values in prec_rec.items():
        precision = pr_values[0]
        recall = pr_values[1]

        ax.step(recall, precision, label=clazz, alpha=.8, where='post')
        # ax.fill_between(recall, precision, alpha=.2, **{'step': 'post'})

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title('Precision Recall Curve')  # TODO: Try to put here the average precision.
    legend = ax.legend(loc='best', bbox_to_anchor=(1.05, 1))
    if fig_path is not None:
        plt.savefig(fig_path, bbox_extra_artists=(legend,), bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.clf()

