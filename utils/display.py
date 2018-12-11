from matplotlib import pyplot as plt
import numpy as np

def show_image(img_data):
    plt.imshow(img_data.T, interpolation='nearest', cmap="gray")
    plt.title("Mean of patch: {:.2f}".format(np.mean(img_data)))
    plt.show()
