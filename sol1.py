import numpy as np
import imageio
from skimage import color
import matplotlib.pyplot as plt

GRAY = 1
COLOR = 2
RGB_TO_YIQ_MATRIX = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
YIQ_TO_RGB_MATRIX = np.linalg.inv(RGB_TO_YIQ_MATRIX)


def read_image(filename, representation):
    """
    Reads an image
    :param filename: path to image file
    :param representation: controls convertion of output: 1 for grayscale output or 2 for RGB output
    :return: np.ndarray, type float64, normalized to [0,1]
    """
    image = imageio.imread(filename)
    if representation == GRAY:
        return color.rgb2gray(image).astype(np.float64)
    return (image / 255).astype(np.float64)


def imdisplay(filename, representation):
    """
    displays an image on the screen.
    :param filename: path to image file
    :param representation: controls output: 1 for grayscale output or 2 for RGB output
    :return: None
    """
    image = read_image(filename, representation)
    if representation == GRAY:
        image = color.rgb2gray(image).astype(np.float64)
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(image)
    plt.show()


def rgb2yiq(imRGB):
    """
    converts a colored image from RGB representation to YIQ
    :param imRGB: np.ndarray (image to be converted)
    :return: np.ndarray (converted image)
    """
    return np.dot(imRGB, RGB_TO_YIQ_MATRIX.transpose())


def yiq2rgb(imYIQ):
    """
    converts a colored image from YIQ representation to RGB

    :param imYIQ: np.ndarray (image to be converted)
    :return: np.ndarray (converted image)
    """
    return np.dot(imYIQ, YIQ_TO_RGB_MATRIX.transpose())


def histogram_equalize(im_orig):
    """
     Performs histogram equalization on an image (and doesn't change the original image).
    :param im_orig: np.ndarray - image to do histogram equalization on
    :return: np.ndarray of image after completing equalization
    """
    gray_image = im_orig.copy()
    image = im_orig.copy()
    if len(im_orig.shape) == 3:
        image = rgb2yiq(image)
        gray_image = image[:, :, 0]
    gray_image_full_scale = np.round(gray_image * 255).astype(int)
    hist, edges = np.histogram(gray_image_full_scale.flatten(), bins=range(257))
    c_hist = np.cumsum(hist)
    m, q = int(gray_image_full_scale.min()), int(gray_image_full_scale.max())
    if c_hist[m] == c_hist[q]:
        return gray_image
    lut = np.round(255 * (c_hist - c_hist[m]) / (c_hist[q] - c_hist[m]))
    if len(im_orig.shape) == 3:
        image[:, :, 0] = lut[gray_image_full_scale] / 255
        return yiq2rgb(image), hist, c_hist
    return lut[gray_image_full_scale] / 255, hist, c_hist


def quantize(im_orig, n_quant, n_iter):
    """
    performs quantization to an image (without changing original image)
    :param im_orig: np.ndarray - original image which we want to quantize
    :param n_quant: number of colors to reduce to
    :param n_iter: maximum iterations to perform in the iterative algorithm
    :return: np.ndarray of image after completing quantization
    """
    gray_image = im_orig.copy()
    image = im_orig.copy()
    if len(im_orig.shape) == 3:  # if image is RGB
        image = rgb2yiq(im_orig.copy())
        gray_image = image[:, :, 0]
    gray_image_full_scale = np.round(gray_image * 255).astype(int)
    hist, edges = np.histogram(gray_image_full_scale.flatten(), bins=range(257))
    c_hist = np.cumsum(hist)
    z = init_z(gray_image_full_scale, n_quant, c_hist)
    error_arr = []
    for iteration in range(n_iter):
        prev_z = z.copy()
        # print(f"iteration number {iteration}")
        q = compute_q(z, n_quant, hist)
        z = compute_z(q, n_quant)
        # compute error
        error_arr.insert(len(error_arr), compute_error(z, q, hist, n_quant))
        if np.array_equal(prev_z, z):
            break

    prcsd_image_grayscale = gray_image_full_scale.copy()
    for i in range(n_quant):
        prcsd_image_grayscale[(prcsd_image_grayscale > z[i]) & (prcsd_image_grayscale <= z[i + 1])] = np.round(q[i])
    if len(im_orig.shape) == 3:
        image[:, :, 0] = prcsd_image_grayscale / 255
        return yiq2rgb(image), error_arr
    return prcsd_image_grayscale / 255, np.array(error_arr)


def compute_q(z_array, n_quant, hist):
    """
    computes q array (of colors to reduce to) for a given z array (edges) according to quantization algorithm
    :param z_array: np.ndarray of shape (n_quant + 1,) , with the current edges
    :param n_quant: number of colors to reduce to by quantization
    :param hist: the histogram of the given image
    :return: np.ndarray of shape (n_quant,) - updated q array according to the z array.
    """
    q_ret = np.zeros(n_quant)
    for i in range(n_quant):
        if np.floor(z_array[i]) == np.floor(z_array[i + 1]):
            q_ret[i] = z_array[i + 1]
        else:
            bot_idx = int(np.floor(z_array[i]) + 1)
            top_idx = int(np.floor(z_array[i + 1]))
            if np.sum(hist[bot_idx:top_idx + 1]) == 0:
                q_ret[i] = z_array[i+1]
            else:
                q_ret[i] = np.dot(hist[bot_idx:top_idx + 1], np.array(range(bot_idx, top_idx + 1))) / \
                       np.sum(hist[bot_idx:top_idx + 1])
    return q_ret


def compute_z(q_arr, n_quant):
    """
    computes z array (of edges) for a given q array (colors to reduce to) according to quantization algorithm
    :param q_arr: np.ndarray of shape (n_quant,), representing the colors
    :param n_quant: number of colors to reduce to by quantization
    :return: np.ndarray of shape (n_quant + 1,), representing the edges for the given q.
    """
    z_ret = np.zeros(n_quant + 1)
    z_ret[1:-1] = (q_arr[:-1] + q_arr[1:]) / 2
    z_ret[0] = -1
    z_ret[-1] = 255
    return z_ret


def init_z(image, n_quant, c_hist):
    """
    initializes z array for quantization algorithm,
    so that there are approximately the same amount of pixels in every range.
    :param image: np.ndarray of given image to do quantization on
    :param n_quant: number of colors for quantization
    :param c_hist: histogram of the given image
    :return: np.ndarray of shape (n_quant + 1,) with initialized values.
    """
    ret = np.zeros(n_quant + 1)
    for i in range(n_quant):
        ret[i] = np.argmax(c_hist >= i * (image.shape[0] * image.shape[1] / n_quant))
    ret[0] = -1
    ret[n_quant] = 255
    return ret


def compute_error(z, q, hist, n_quant):
    """
    computes the error for a given z, q as in the quantization algorithm.
    :param z: np.ndarray of shape (n_quant + 1,) , with the current edges
    :param q: np.ndarray of shape (n_quant,), representing the colors chosen for quantization
    :param hist: histogram of the image
    :param n_quant: number of colors
    :return: computed error
    """
    array_to_sum = np.zeros(n_quant)
    for i in range(n_quant):
        bot_idx = int(np.floor(z[i]) + 1)
        top_idx = int(np.floor(z[i + 1]))
        g = np.array(range(bot_idx, top_idx + 1))
        qi_minus_g_squared = (q[i] - g) ** 2
        array_to_sum[i] = np.dot(qi_minus_g_squared, hist[bot_idx:top_idx + 1])
    return np.sum(array_to_sum)


if __name__ == '__main__':
    # toy picture:
    x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :], np.array([255] * 6)[None, :]])
    grad = np.tile(x, (256, 1))
    # y, hist1, hist2 = histogram_equalize(grad / 255)
    # plt.imshow(y, cmap='gray')
    # plt.show()
    # z_quant, array_of_errors = quantize(y, 9, 10)
    # plt.imshow(z_quant, cmap='gray')
    # plt.show()
    lenna_image = read_image("C:\\Users\\danie\\Desktop\\Uni\\YearD\\"
                             "Image_Processing\\ex1-azulay93\\impr_ex1_helper_resources\\Starry_Night.jpg", 1)
    lenna_eq, hist_a, hist_b = histogram_equalize(lenna_image)
    lenna_q, errors_array = quantize(lenna_image, 5, 100)
    plt.imshow(lenna_q)
    plt.show()
    plt.imshow(lenna_eq)
    plt.show()
