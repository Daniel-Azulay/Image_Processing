##################################################
#        HUJI Image Processing 2022-2023         #
#                  Ex1 Helper                    #
#                                                #
#      Originally from 2021-2022 by orelby       #
#                                                #
#        Last updated at 2022-11-09 21:15        #
##################################################

"""
Runs histogram equalization, quantization and RGB quantization on selected images,
groups them, and displays and/or saves the results.

Instructions:
1. Place your `sol1.py` in a directory along with this file and its resource directory.
2. (Optional) See "Basic Configuration" below (e.g. Choose not to view or save results).
3. (Optional) See "Task Configuration" below (e.g. Toggle/change the tasks to run).
2. Run this python file.
   Running from PyCharm will display the results in SciView.
   Running from the command line might not display the results,
   or pause the program whenever a new image is displayed until you close it.
"""

import sol1

import math
import os
import itertools as it
import warnings

import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread

import matplotlib

# Should work fine in PyCharm's SciView, idk :)
# Look into `matplotlib.use` if needed
warnings.filterwarnings(
    "ignore", category=matplotlib.MatplotlibDeprecationWarning,
    message='^Support for FigureCanvases without a required_interactive_framework'
)

##################################################
#              Basic Configuration               #
##################################################

# Show results in a GUI if supported (be aware that figures may be quite large)
SHOULD_SHOW = True

# Save results as files (will be saved under a new directory called results_{timestamp})
SHOULD_SAVE = True

# Warn if quantization error is not weakly monotonically decreasing
SHOULD_CHECK_Q_ERROR = True

# Source directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
RESOURCE_DIR = BASE_DIR + 'impr_ex1_helper_resources' + os.sep

##################################################
#               Task Configuration               #
##################################################

# To only run histogram equalization / quantization / RGB quantization
# comment on/off the relevant list items in `run()`

# To change the image sets see the following constants
# and the parameters of each runner function bellow.
# e.g. `runner_histogram_equalization(im_orig, im_expected=None)` runs on `TASKS_HEQ` items
#  (its output is passed to `collect_figure(...)` through `run_task_set(...)`)

TASKS_HEQ = [
    [
        np.array(
            [[52, 55, 61, 59, 79, 61, 76, 61, ],
             [62, 59, 55, 104, 94, 85, 59, 71, ],
             [63, 65, 66, 113, 144, 104, 63, 72, ],
             [64, 70, 70, 126, 154, 109, 71, 69, ],
             [67, 73, 68, 106, 122, 88, 68, 68, ],
             [68, 79, 60, 70, 77, 66, 58, 75, ],
             [69, 85, 64, 58, 55, 61, 65, 83, ],
             [70, 87, 69, 68, 65, 73, 78, 90, ]]
        ) / 255,
        np.array(
            [[0, 12, 53, 32, 190, 53, 174, 53, ],
             [57, 32, 12, 227, 219, 202, 32, 154, ],
             [65, 85, 93, 239, 251, 227, 65, 158, ],
             [73, 146, 146, 247, 255, 235, 154, 130, ],
             [97, 166, 117, 231, 243, 210, 117, 117, ],
             [117, 190, 36, 146, 178, 93, 20, 170, ],
             [130, 202, 73, 20, 12, 53, 85, 194, ],
             [146, 206, 130, 117, 85, 166, 182, 215, ]]
        ) / 255,
    ],
    [
        np.tile(np.hstack([
            np.repeat(np.arange(0, 50, 2), 10)[None, :],
            np.array([255] * 6)[None, :]
        ]), (256, 1)).astype(np.float64) / 255,
    ],
    [
        sol1.read_image(RESOURCE_DIR + 'Unequalized_Hawkes_Bay_NZ.jpg', 1),
        sol1.read_image(RESOURCE_DIR + 'Equalized_Hawkes_Bay_NZ.jpg', 1)
    ],
    [
        sol1.read_image(RESOURCE_DIR + 'low_contrast.jpg', 1)
    ],
    [
        sol1.read_image(RESOURCE_DIR + 'low_contrast.jpg', 2)
    ],
]

TASKS_Q = [
    [
        np.tile(np.hstack([
            np.repeat([0, 10, 21, 31, 41, 52, 62, 73, 83, 93, 104, 114, 124, 135,
                       145, 155, 166, 176, 187, 197, 207, 218, 228, 238, 249], 10)[None, :],
            np.array([255] * 6)[None, :]
        ]), (256, 1)).astype(np.float64) / 255,
        [1, 2, 3, 4, 5, 8, 12, 25]
    ],
    [
        sol1.read_image(RESOURCE_DIR + 'Equalized_Hawkes_Bay_NZ.jpg', 1),
        [1, 2, 3, 4, 5, 6, 8, 30]
    ],
    [
        sol1.read_image(RESOURCE_DIR + 'jerusalem.jpg', 1),
        [1, 2, 3, 4, 5, 8, 16, 30]
    ],
    [
        imread(RESOURCE_DIR + 'jerusalem.jpg').astype(np.float64) / 255,
        [1, 2, 3, 4, 5, 8, 16, 30]
    ],
    [
        sol1.read_image(RESOURCE_DIR + 'Robert_Duncanson.jpg', 1),
        [1, 2, 3, 5, 8]
    ],
    [
        sol1.read_image(RESOURCE_DIR + 'low_contrast.jpg', 1),
        [1, 2, 3, 4, 5, 6, 8, 30]
    ],
    [
        sol1.histogram_equalize(sol1.read_image(RESOURCE_DIR + 'low_contrast.jpg', 1))[0],
        [1, 2, 3, 4, 5, 6, 8, 30]
    ],
]

TASKS_Q_RGB = [
    [
        sol1.read_image(RESOURCE_DIR + 'monkey.jpg', 2),
        [1, 2, 3, 4, 5, 6, 8, 16, 30, 50, 100]
    ],
    [
        imread(RESOURCE_DIR + 'jerusalem.jpg').astype(np.float64) / 255,
        [1, 2, 3, 4, 5, 6, 8, 16, 30, 50, 200]
    ],
    [
        sol1.read_image(RESOURCE_DIR + 'Robert_Duncanson.jpg', 2),
        [1, 2, 3, 4, 5, 6, 7, 8, 16, 30, 50]
    ],
]

TASKS_Q_RGB_EXTRA = [
    [
        np.tile(
            np.repeat(
                np.hstack([
                    np.repeat([0, 10, 21, 31, 41, 52, 62, 73, 83, 93, 104, 114, 124, 135,
                               145, 155, 166, 176, 187, 197, 207, 218, 228, 238, 249], 10)[None, :],
                    np.array([255] * 6)[None, :]
                ]),
                3
            ).reshape((-1, 3)),
            (256, 1, 1)
        ).astype(np.float64) / 255,
        [1, 2, 3, 4, 5, 8, 12, 25]
    ],
    [
        # Taken from Wiki's Rainbow_Flag.png
        np.repeat(
            np.repeat(
                np.array([
                    [228, 3, 3], [242, 71, 2], [255, 140, 0], [255, 145, 0], [255, 233, 0],
                    [255, 237, 0], [128, 183, 19], [0, 128, 38], [0, 125, 50], [0, 78, 251],
                    [0, 77, 255], [58, 42, 195], [117, 7, 135]
                ]),
                480,
                axis=0
            ).reshape((-1, 480, 3)),
            [49, 1, 48, 1, 1, 48, 1, 48, 1, 1, 48, 1, 49],
            axis=0
        ).astype(np.float64) / 255,
        [1, 2, 3, 4, 5, 6, 7, 8, 15, 25, 50]
    ],
    [
        sol1.read_image(RESOURCE_DIR + 'Starry_Night.jpg', 2),
        [1, 2, 3, 4, 5, 6, 7, 8, 16, 30, 50]
    ],
    [
        sol1.read_image(RESOURCE_DIR + 'this_is_fine.jpeg', 2),
        [100, 25, 16, 8, 4, 3, 2, 1]
    ],
]

INTENSITIES = np.arange(0, 256)


def runner_histogram_equalization(im_orig, im_expected=None):
    im_result, hist_orig, hist_eq = sol1.histogram_equalize(im_orig)

    titles = ['Original', 'Expected After HEQ', 'Result After HEQ']
    images = (im_orig, im_expected, im_result)

    # def _plot_hists(ax):
    #     ax.fill_between(np.arange(256), np.cumsum(hist_orig), color='b')
    #     ax.fill_between(np.arange(256), np.cumsum(hist_eq), color='g')
    #     ax.set_axis_on()
    #
    # titles = ['Original', 'Expected After HEQ', 'Result After HEQ', 'Histograms']
    # images = (im_orig, im_expected, im_result, _plot_hists)

    items = [
        (title, im) for title, im
        in zip(titles, images)
        if im is not None
    ]

    # Pycharm:
    #  "Remove redundant parentheses"
    # Also Pycharm:
    #  "Python version 3.7 does not support unpacking without parentheses in return statements"
    return (len(items), *(im_orig.shape[:2]), items)


def runner_quantization(im_orig, n_quants, n_iter=50, should_check_error=SHOULD_CHECK_Q_ERROR):
    def _q_generator():
        for n_quant in n_quants:
            im_q, error = sol1.quantize(im_orig, n_quant, n_iter)

            if should_check_error and np.any(np.diff(error) > 0):
                warnings.warn('Found an increase in quantization error for Q = ' + str(n_quant)
                              + ' (performed ' + str(len(error)) + ' iterations).')

            yield im_q

    total_items = len(n_quants) + 1
    titles = it.chain(('Original',), ('Result After Q = ' + str(n_quant) for n_quant in n_quants))
    images = it.chain((im_orig,), _q_generator())

    return (total_items, *(im_orig.shape[:2]), zip(titles, images))


def runner_rgb_quantization(im_orig, n_quants):
    total_items = len(n_quants) + 1
    titles = it.chain(('Original',),
                      ('Result After RGBQ = ' + str(n_quant) for n_quant in n_quants))
    images = it.chain((im_orig,),
                      (sol1.quantize_rgb(im_orig, n_quant) for n_quant in n_quants))

    return (total_items, *(im_orig.shape[:2]), zip(titles, images))


# TODO: check/show histograms, quantification errors and iteration count
# TODO: accept it's not going to happen and remove last TODO

def collect_figure(total_subplots, subplot_height, subplot_width, items,
                   should_show=SHOULD_SHOW, filename=None):
    rows = math.floor(math.sqrt(total_subplots))
    cols = math.ceil(total_subplots / rows)

    # Nothing to see here, just some random HUJI magic heuristics

    if cols > rows and 3 * subplot_width >= 4 * subplot_height:
        rows, cols = cols, rows
    elif rows > cols and 3 * subplot_width <= 4 * subplot_height:
        rows, cols = cols, rows

    width = max(6.4, cols * (subplot_width / 100) + 1)
    height = max(3, rows * (subplot_height / 100) + 1)
    fontsize = math.ceil(max(width, height))

    fig, axs = plt.subplots(rows, cols, figsize=(width, height))
    axs = axs.flatten()

    for ax, (title, im) in it.zip_longest(axs, items, fillvalue=(None, None)):
        ax.set_axis_off()

        if title is not None:
            ax.set_title(title, fontsize=fontsize)

        if callable(im):
            im(ax)
        elif im is not None:
            ax.imshow(im, cmap='gray' if im.ndim == 2 else None, vmin=0, vmax=1)

    plt.tight_layout()

    if isinstance(filename, str):
        plt.savefig(filename + '.jpg')

    if should_show:
        plt.show()


def run_task_set(task_set, resolver, should_show=SHOULD_SHOW, save_prefix=None):
    for i, args in enumerate(task_set):
        print('- Running ' + str(i) + '...')
        collect_figure(*resolver(*args),
                       should_show,
                       save_prefix + str(i) if save_prefix is not None else None)


def run_task_sets(task_sets, should_show=SHOULD_SHOW, should_save=SHOULD_SAVE):
    save_dir = None

    if should_show:
        print('Output will be shown in a GUI if supported (as configured).'
              + ' Be aware that the figures may be quite large.')
    else:
        print('Output will not be shown in a GUI (as configured).')

    if should_save:
        import time

        save_dir = BASE_DIR + 'results_' + time.strftime('%Y-%m-%d_%H-%M-%S')

        if os.path.exists(save_dir):
            raise Exception('Output directory path ' + save_dir + ' already exists.')

        print('Creating output directory "' + save_dir + '"...')
        os.mkdir(save_dir)
        print('Created output directory.')

    else:
        print('Output will not be saved (as configured).')

    for (task_set, resolver, save_prefix) in task_sets:
        print('Running: ' + save_prefix.upper())
        save_prefix = save_dir + os.sep + save_prefix if should_save else None
        run_task_set(task_set, resolver, should_show, save_prefix)


def run():
    all_task_sets = [
        (TASKS_HEQ, runner_histogram_equalization, 'heq'),
        (TASKS_Q, runner_quantization, 'q')
    ]

    if 'quantize_rgb' in dir(sol1):
        all_task_sets.extend([
            (TASKS_Q_RGB, runner_rgb_quantization, 'rgbq'),
            (TASKS_Q_RGB_EXTRA, runner_rgb_quantization, 'rgbq_extra'),
        ])

    print('Starting...')

    run_task_sets(all_task_sets)

    print('Finished')


if __name__ == "__main__":
    run()
