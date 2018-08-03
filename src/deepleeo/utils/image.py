import numpy as np
import pylab as plt

def plot_image_histogram(raster_array, colors=None, nbins = 256, title="Histogram"):
    fig = plt.figure(figsize=(12, 8))
    plt.title(title, fontsize=15)
    plt.ylim([0,100000])
    plt.xlabel("Bins", fontsize=12)
    plt.ylabel("Number of Pixels", fontsize=12)

    if(colors is None):
        colors = plt.cm.get_cmap("hsv", raster_array.shape[2])

    for band in range(raster_array.shape[2]):
        ax_data = np.reshape(raster_array[:,:,band],-1)
        ax_color = colors(band)
        plt.hist(ax_data, nbins, color=ax_color, alpha=0.4)

    plt.show()