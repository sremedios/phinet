from matplotlib import pyplot as plt

def show_image(img_data):
    plt.imshow(img_data, interpolation='nearest', cmap="Greys")
    plt.show()
