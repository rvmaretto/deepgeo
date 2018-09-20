import skimage
import pylab as plt
import seaborn as sns
import numpy as np
from skimage import exposure

def plot_rgb_img(raster_array, bands=[1,2,3], contrast=False, title="RGB Composition"):
    if len(bands) != 3 and len(bands) != 1:
        raise AttributeError("Parameter bands must have size 3 or 1.")

    plt.figure(figsize=(10, 10))
    plt.title(title)
    raster_img = skimage.img_as_ubyte(raster_array)
    if contrast:
        for band in bands:
            p2, p98 = np.percentile(raster_img[:, :, band], (2, 98))
            raster_img[:, :, band] = exposure.rescale_intensity(raster_img[:, :, band], in_range=(p2, p98))

    if len(bands) == 3:
        plt.imshow(raster_img[:, :, bands])
    else:
        plt.imshow(raster_img[:,:,bands[0]])
    plt.axis('off')

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