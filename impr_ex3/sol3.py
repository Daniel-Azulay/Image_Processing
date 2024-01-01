import numpy as np
from scipy import ndimage
import imageio
from skimage import color
import matplotlib.pyplot as plt
import os

MIN_IM_DIM = 16


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


def expand(im, blur_filter):
    """
    Expand an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the expanded image
    """
    expanded_image = np.zeros((2 * im.shape[0], 2 * im.shape[1]))
    expanded_image[::2, ::2] = im
    expanded_image = ndimage.convolve(expanded_image, 2 * blur_filter)
    expanded_image = ndimage.convolve(expanded_image, np.transpose(2 * blur_filter))
    return expanded_image


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


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
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
    gaussian_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    n = len(gaussian_pyr) - 1
    l_pyr = []
    for k in range(n):
        l_pyr.append(gaussian_pyr[k] - expand(gaussian_pyr[k + 1], filter_vec))
    l_pyr.append(gaussian_pyr[n])
    return l_pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: Laplacian pyramid
    :param filter_vec: Filter vector
    :param coeff: A python list in the same length as the number of levels in
            the pyramid lpyr.
    :return: Reconstructed image
    """
    n = len(lpyr) - 1
    cur_image = lpyr[n] * coeff[n]
    for i in range(n - 1, -1, -1):
        cur_image = (lpyr[i] * coeff[i]) + expand(cur_image, filter_vec)
    return cur_image


def read_image(filename, representation):
    """
    Reads an image
    :param filename: path to image file
    :param representation: controls convertion of output: 1 for grayscale output or 2 for RGB output
    :return: np.ndarray, type float64, normalized to [0,1]
    """
    image = imageio.imread(filename)
    if representation == 1:
        if len(image.shape) >= 3 and image.shape[2] >= 4:
            image = image[:, :, :3]
        if len(image.shape) >= 3:
            return color.rgb2gray(image).astype(np.float64)
    return (image / 255).astype(np.float64)


def render_pyramid(pyr, levels):
    """
    Render the pyramids as one large image with 'levels' smaller images
        from the pyramid
    :param pyr: The pyramid, either Gaussian or Laplacian
    :param levels: the number of levels to present
    :return: res a single black image in which the pyramid levels of the
            given pyramid pyr are stacked horizontally.
    """
    pyr_copy = pyr.copy()
    for j in range(len(pyr)):
        pyr_copy[j] = pyr_copy[j] - pyr_copy[j].min()
        pyr_copy[j] = pyr_copy[j] / pyr_copy[j].max()
    big_image = pyr_copy[0]
    for i in range(1, levels):
        rows_of_zero = ((big_image.shape[0] // pyr_copy[i].shape[0]) - 1) * pyr_copy[i].shape[0]
        pic_of_zeros = np.zeros((rows_of_zero, pyr_copy[i].shape[1]))
        pic_to_stack = np.vstack((pyr_copy[i], pic_of_zeros))
        big_image = np.hstack((big_image, pic_to_stack))
    return big_image


def display_pyramid(pyr, levels):
    """
    display the rendered pyramid
    """
    im_to_display = render_pyramid(pyr, levels)
    plt.imshow(im_to_display, cmap='gray', vmin=0, vmax=1)
    plt.show()
    return


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
     Pyramid blending implementation
    :param im1: input grayscale image
    :param im2: input grayscale image
    :param mask: a boolean mask
    :param max_levels: max_levels for the pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd
            scalar that represents a squared filter)
    :param filter_size_mask: size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask
    :return: the blended image
    """
    lap_pyr1, lap_vec1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lap_pyr2, lap_vec2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    mask_gaus_pyr, mask_gaus_vec = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    l_out = []
    masked_pyr_1 = []
    masked_pyr_2 = []
    for k in range(len(lap_pyr1)):
        masked_lap_1 = np.multiply(mask_gaus_pyr[k], lap_pyr1[k])
        masked_pyr_1.append(masked_lap_1)
        masked_lap_2 = np.multiply(np.ones(mask_gaus_pyr[k].shape) - mask_gaus_pyr[k], lap_pyr2[k])
        masked_pyr_2.append(masked_lap_2)
        l_out.append(masked_lap_1 + masked_lap_2)
    # plt.imshow(laplacian_to_image(masked_pyr_1, lap_vec1, np.ones(len(masked_pyr_1))), cmap='gray', vmin=0, vmax=1)
    # plt.show()
    # plt.imshow(laplacian_to_image(masked_pyr_2, lap_vec2, np.ones(len(masked_pyr_1))), cmap='gray', vmin=0, vmax=1)
    # plt.show()
    # plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
    # plt.show()
    result = laplacian_to_image(l_out, lap_vec1, np.ones(len(l_out)))
    result = np.clip(result, 0, 1)
    return result


def blending_example1():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    image1_name = "bibi.jpg"
    image2_name = "avi_maoz.png"
    mask_name = "bibi_mask.jpg"
    img_1 = read_image(relpath('externals/' + image1_name), 2)
    img_2 = read_image(relpath('externals/' + image2_name), 2)
    blended_color_image = np.zeros(img_1.shape)
    mask = read_image(relpath('externals/' + mask_name), 1).astype(bool)
    for i in range(3):
        blended_color_image[:, :, i] = pyramid_blending(img_1[:, :, i], img_2[:, :, i], mask, 5, 5, 5)
    display_four_images(img_1, img_2, mask, blended_color_image)
    return img_1, img_2, mask, blended_color_image


def blending_example2():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    image1_name = "peleg.jpg"
    image2_name = "lottery.jpg"
    mask_name = "peleg_mask.jpg"
    img_1 = read_image(relpath('externals/' + image1_name), 2)
    img_2 = read_image(relpath('externals/' + image2_name), 2)
    blended_color_image = np.zeros(img_1.shape)
    mask = read_image(relpath('externals/' + mask_name), 1).astype(np.bool)
    for i in range(3):
        blended_color_image[:, :, i] = pyramid_blending(img_1[:, :, i], img_2[:, :, i], mask, 5, 5, 5)
    display_four_images(img_1, img_2, mask, blended_color_image)
    return img_1, img_2, mask, blended_color_image


def display_four_images(img1, img2, mask, blended_img):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(img1)
    axs[0, 1].imshow(img2)
    axs[1, 0].imshow(mask, cmap='gray', vmin=0, vmax=1)  # mask is shown in black and white
    axs[1, 1].imshow(blended_img)
    plt.show()
    return


def pad_to_next_exp_size(im):
    x_exp = 0
    y_exp = 0
    while (2 ** x_exp) < im.shape[0]:
        x_exp += 1
    while (2 ** y_exp) < im.shape[1]:
        y_exp += 1
    x_pad_num = ((2 ** x_exp)-im.shape[0]) // 2
    y_pad_num = ((2 ** y_exp)-im.shape[1]) // 2
    return np.pad(im, ((x_pad_num, x_pad_num), (y_pad_num, y_pad_num)))


def crop_to_exp_size(im):
    x_exp = 0
    y_exp = 0
    while (2 ** x_exp) < im.shape[0]:
        x_exp += 1
    while (2 ** y_exp) < im.shape[1]:
        y_exp += 1
    return im[:2 ** (x_exp - 1), :2 ** (y_exp - 1)]


def make_binary_gray_to_black(mask_name):
    mask = read_image(relpath('externals/' + mask_name), 1)
    mask[mask < 1] = 0
    return mask


def make_binary_gray_to_white(mask_name):
    binary_mask = read_image(relpath('externals/' + mask_name), 1)
    binary_mask = np.where(binary_mask > 0, 1, 0)
    imageio.imwrite(relpath('externals/' + 'bool_' + mask_name), binary_mask)
    return


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


if __name__ == '__main__':

    # im1, im2, mask_image, blended = blending_example1()
    # plt.imshow(blended)
    # plt.show()

    im1, im2, mask_image, blended = blending_example1()
    im1, im2, mask_image, blended = blending_example2()

    # gaus_pyr, gaus_vec = build_gaussian_pyramid(image_to_process, 9, 3)
    # display_pyramid(gaus_pyr, len(gaus_pyr))
    # lap_pyr, lap_vec = build_laplacian_pyramid(image_to_process, 9, 3)
    # display_pyramid(lap_pyr, len(lap_pyr))
    # image_reconstructed = laplacian_to_image(lap_pyr, lap_vec, np.ones(len(lap_pyr)))
    # plt.imshow(np.hstack((image_reconstructed, image_to_process)), cmap='gray', vmin=0, vmax=1)
    # print(not(False in np.isclose(image_reconstructed, image_to_process)))
    # print(np.max(np.abs(image_reconstructed - image_to_process)))
    # plt.show()
    # for i in range(1, len(pyr)):
    #     plt.subplot(1, len(pyr), i)
    #     plt.imshow(lap_pyr[i-1], cmap='gray')
    # plt.show()
    # plt.imshow(sum_lap, cmap='gray')
    # plt.show()
    # im_to_display = expand(image[::2, ::2])
    # plt.imshow(im_to_display, cmap='gray')
    # plt.show()
