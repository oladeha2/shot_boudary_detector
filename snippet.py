import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt


def displaySnippet(snippet):
    """
    Display ten images in a 1x10 structure sublot figure. Figsize for each subplot is 80x60
    Args:
        Ten images in the form of numpy arrays
    """
    figure = plt.figure(figsize=(100,90))
    for idx, frame in enumerate(snippet):
        figure.add_subplot(1,len(snippet), idx+1).imshow(frame, interpolation='nearest')
    plt.show()


def getSnippet(image_paths, start, end):
    """
    Return an array of ten nunpy images normalized 
    Args:
        list of image paths
        start and end indexes for snippet selection
        frames are normalized before being returned
    """
    return [np.array(Image.open(image_paths[i])) for i in range(start, end)]