from matplotlib import pyplot as plt
import numpy as np

def show_image(img_data, img_class=None):
    plt.imshow(img_data.T, interpolation='nearest', cmap="gray")
    if img_class:
        plt.title("Class: {}    Mean of patch: {:.2f}".format(img_class, np.mean(img_data)))
    else:
        plt.title("Mean of patch: {:.2f}".format(np.mean(img_data)))
    plt.show()
