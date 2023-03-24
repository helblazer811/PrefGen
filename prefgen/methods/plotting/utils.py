import torch
import numpy as np
import traceback

def convert_stylegan_image_to_matplotlib(image):
    """
        Converts an image from the stylegan format to a valid
        image to be displayed with matplotlib imshow 
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if len(image.shape) > 3:
        image = image.squeeze()
    image = np.transpose(image, (1, 2, 0))
    image = (image - np.min(image)) / np.max(image - np.min(image))
    image *= 255
    image = image.astype(int)
    return image 

def plot_ignore_exceptions(f):
    """
        Wrapper function for handling exceptions while
        still returning the errors. 
    """
    def wrapper(*args, **kw):
        try:
            return f(*args, **kw)
        except Exception as exception:
            print(exception)
            print(traceback.format_exc())

    return wrapper