from scipy.signal import convolve2d
import numpy as np
import imageio
from skimage import color
from scipy import ndimage

MIN_IM_DIM = 16


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


def read_image(filename, representation):
    """
    Reads an image
    :param filename: path to image file
    :param representation: controls convertion of output: 1 for grayscale output or 2 for RGB output
    :return: np.ndarray, type float64, normalized to [0,1]
    """
    image = imageio.imread(filename)
    if representation == 1:
        return color.rgb2gray(image).astype(np.float64)
    return (image / 255).astype(np.float64)


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    gaussian_pyr = [im.copy()]
    cur_image = im.copy()
    filter_vec = np.array([1])
    for i in range(filter_size - 1):
        filter_vec = np.convolve(filter_vec, [1, 1])
        filter_vec = filter_vec / np.sum(filter_vec)
    filter_vec = np.reshape(filter_vec, (1, len(filter_vec)))
    for t in range(max_levels - 1):
        cur_image = reduce(cur_image, filter_vec)
        if (cur_image.shape[0] < MIN_IM_DIM) | (cur_image.shape[1] < MIN_IM_DIM):
            break
        gaussian_pyr.append(cur_image)
    return gaussian_pyr, filter_vec


def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    # blur:
    cur_image = ndimage.convolve(im, blur_filter)
    cur_image = ndimage.convolve(cur_image, np.transpose(blur_filter))

    # sample:
    return cur_image[::2, ::2]