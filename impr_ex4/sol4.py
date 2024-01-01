# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged 

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
import shutil
from imageio import imwrite
import itertools as it

import sol4_utils

VERTICAL_RECTANGLES, HORIZONTAL_RECTANGLES = (7, 7)
HARRIS_CORNER_K = 0.04
PATCH_SIZE_K = 7
HARRIS_CORNER_BOUNDRY_GAP = 15
MIN_SCORE = 0.9



def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    # calculate derivatives:
    x_der_filter = np.array([[1, 0, -1]])
    y_der_filter = np.transpose(x_der_filter)
    der_x = ndimage.convolve(im, x_der_filter)
    der_y = ndimage.convolve(im, y_der_filter)

    # calculate r:
    squared_der_x = sol4_utils.blur_spatial(np.multiply(der_x, der_x), kernel_size=3)
    squared_der_y = sol4_utils.blur_spatial(np.multiply(der_y, der_y), kernel_size=3)
    der_x_der_y = sol4_utils.blur_spatial(np.multiply(der_x, der_y), kernel_size=3)
    det_m = np.multiply(squared_der_x, squared_der_y) - np.square(der_x_der_y)
    trace_m = squared_der_x + squared_der_y
    r = det_m - HARRIS_CORNER_K * np.square(trace_m)

    # return indices where non_maximum_suppression is not 0, flipped for (x,y) instead of (row,column):
    return np.flip(np.argwhere(non_maximum_suppression(r)))


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    k = 2 * desc_rad + 1
    patches = np.zeros((pos.shape[0], k, k))
    for i, p in enumerate(pos):
        # coordinates of window KxK around the point p:
        x_range = np.arange(p[0] - desc_rad, p[0] + desc_rad + 1)
        y_range = np.arange(p[1] - desc_rad, p[1] + desc_rad + 1)
        (col_coordinates, row_coordinates) = np.meshgrid(x_range, y_range)

        # calculate patch around point p using map_coordinates:
        # (switch columns because map_coordinates accepts coordinates as (rows, cols)
        patch = ndimage.map_coordinates(im, [row_coordinates, col_coordinates], order=1, prefilter=False)

        # normalize:
        patch_mean = np.mean(patch)
        norm = np.linalg.norm(patch - patch_mean)
        if norm == 0:
            patch = np.zeros((k, k))
        else:
            patch = (patch.reshape((k, k)) - patch_mean) / norm
        patches[i] = patch
    return patches


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """
    feature_points = spread_out_corners(pyr[0], VERTICAL_RECTANGLES, HORIZONTAL_RECTANGLES, HARRIS_CORNER_BOUNDRY_GAP)
    feature_points_level_2 = (1 / 4) * feature_points
    return feature_points, sample_descriptor(pyr[2], feature_points_level_2, PATCH_SIZE_K // 2)


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """

    k = desc1.shape[1]

    # flatten every matrix in desc1, desc2:
    flattened_matrices1 = desc1.reshape((desc1.shape[0], k ** 2))
    flattened_matrices2 = desc2.reshape((desc2.shape[0], k ** 2))

    # product of all pairs of descriptors:
    matrix_s = np.matmul(flattened_matrices1, np.transpose(flattened_matrices2))
    matrix_s = matrix_s.reshape((desc1.shape[0], desc2.shape[0]))

    # for each row and then for each columns:
    # leave only 2 highest values (everything else is 0), and substract min_score:
    # can do better without sort??
    rows_2_maximal_values = matrix_s * (matrix_s >= np.sort(matrix_s, axis=1)[:, [-2]]).astype(int) - min_score
    cols_2_maximal_values = matrix_s * (matrix_s >= np.sort(matrix_s, axis=0)[[-2], :]).astype(int) - min_score

    # use only indices that are positive in both matrices
    # (after substracting min_score it means to take best 2 if they are above min_score)
    indices = np.argwhere((rows_2_maximal_values > 0) & (cols_2_maximal_values > 0))

    # seperate the indices to desc1 and desc2:
    desc1_indices = np.array([p[0] for p in indices])
    desc2_indices = np.array([p[1] for p in indices])
    return desc1_indices, desc2_indices


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """

    # add third coordinate for each point:
    homogenious_coords = np.hstack((pos1, np.ones((pos1.shape[0], 1))))

    # transpose for points as columns, compute homography on homogenious coords, and back to points as lines:
    homographied_trios = H12 @ np.transpose(homogenious_coords)
    homographied_trios = np.transpose(homographied_trios)

    # devide in each row the first two columns by the third column:
    new_coords = np.divide(homographied_trios[:, :2], homographied_trios[:, 2][:, None])
    return new_coords


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    amount_of_matches = 2
    size_of_inliers = 0
    inliers_indices = np.array([])
    for i in range(num_iter):

        # choose 1 or 2 pairs of points (choose indexes for the points)
        idxs = np.random.choice(len(points1), amount_of_matches)

        # get the transformation and apply on points1:
        trans_matrix = estimate_rigid_transform(points1[idxs], points2[idxs], translation_only)
        points1_new_coordinates = apply_homography(points1, trans_matrix)

        # decide inliers and update if "broke a record":
        squared_distances = np.linalg.norm(points2 - points1_new_coordinates, axis=1) ** 2
        are_inliers = np.where(squared_distances < inlier_tol, 1, 0)
        num_of_inliers_found = np.sum(are_inliers)
        if num_of_inliers_found > size_of_inliers:
            inliers_indices = np.nonzero(are_inliers)
            size_of_inliers = num_of_inliers_found
    trans_matrix =  estimate_rigid_transform(points1[inliers_indices], points2[inliers_indices], translation_only)
    return trans_matrix / trans_matrix[2, 2], np.array(inliers_indices).reshape((-1,))


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    # stack images:
    plt.imshow(np.hstack(np.array([im1, im2])), cmap='gray')

    # move points of image2 to the right:
    points2[:, 0] += im1.shape[1]

    # create alternating coordinates between the images for the inliers:
    x_coords_inliers = np.ravel([points1[:, 0][inliers], points2[:, 0][inliers]], 'F')
    y_coords_inliers = np.ravel([points1[:, 1][inliers], points2[:, 1][inliers]], 'F')

    # create alternating coordinates between the images for the outliers:
    outliers = np.delete(np.arange(points1.shape[0]), inliers, 0)
    x_coords_outiers = np.ravel([points1[:, 0][outliers], points2[:, 0][outliers]], 'F')
    y_coords_outiers = np.ravel([points1[:, 1][outliers], points2[:, 1][outliers]], 'F')

    # plot the points and lines:
    plt.plot(x_coords_outiers, y_coords_outiers, color='b', marker='o', markersize=3, lw=0.3, mfc='red')
    plt.plot(x_coords_inliers, y_coords_inliers, color='y', marker='o', markersize=3, lw=0.3, mfc='red')
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    # make list of left side of m, meaning i<=m :

    # build the list from m to 0 recursively:
    res_left = [np.eye(3)]
    for i in range(m-1, -1, -1):
        H_im =  res_left[-1] @ H_succesive[i]
        res_left.append(H_im / H_im[2, 2])

    # reverse to get a list from 0 to m:
    res_left.reverse()

    # extend from m to M recursively:
    res = res_left
    for i in range(m + 1, len(H_succesive) + 1):
        H_im = res[-1] @ np.linalg.inv(H_succesive[i - 1])
        res.append(H_im / H_im[2, 2])
    return res


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    corners_of_image = np.array(list(it.product([0, w-1], [0, h-1])))
    corners_after_homography = apply_homography(corners_of_image, homography)
    min_x = np.min(corners_after_homography[:, 0]).astype(int)
    min_y = np.min(corners_after_homography[:, 1]).astype(int)
    max_x = np.max(corners_after_homography[:, 0]).astype(int)
    max_y = np.max(corners_after_homography[:, 1]).astype(int)
    return np.array([[min_x, min_y], [max_x, max_y]])


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    # get warped image coordinates:
    top_left_corner, bot_right_corner = compute_bounding_box(homography, image.shape[1], image.shape[0])
    x = np.arange(top_left_corner[0], bot_right_corner[0] + 1)
    y = np.arange(top_left_corner[1], bot_right_corner[1] + 1)
    xx, yy = np.meshgrid(x, y)

    # turn it into array with shape (Nx2) (every row is a point) and apply inverse homography:
    points = np.hstack((xx.reshape((-1, 1)), yy.reshape((-1, 1))))
    backwarped_coords = apply_homography(points, np.linalg.inv(homography))

    # switch columns as map_coordinates accepts coordinates as (rows, cols) apply_homography returns [x,y]
    backwarped_coords[:, [1, 0]] = backwarped_coords[:, [0, 1]]
    return ndimage.map_coordinates(image, np.transpose(backwarped_coords), order=1, prefilter=False).reshape(xx.shape)


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images, bonus=False):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.bonus = bonus
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        if self.bonus:
            self.generate_panoramic_images_bonus(number_of_panoramas)
        else:
            self.generate_panoramic_images_normal(number_of_panoramas)

    def generate_panoramic_images_normal(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def generate_panoramic_images_bonus(self, number_of_panoramas):
        """
        The bonus
        :param number_of_panoramas: how many different slices to take from each input image
        """
        pass

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            # pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()


# def relpath(filename):
#     return os.path.join(os.path.dirname(__file__), filename)


# if __name__ == '__main__':
    # image_name1 = "oxford1.jpg"
    # img1 = sol4_utils.read_image(relpath('external/' + image_name1), 1)
    # image_name2 = "oxford2.jpg"
    # img2 = sol4_utils.read_image(relpath('external/' + image_name2), 1)
    # # implot = plt.imshow(img1, cmap='gray')
    # # corners1 = spread_out_corners(img1, VERTICAL_RECTANGLES, HORIZONTAL_RECTANGLES, HARRIS_CORNER_BOUNDRY_GAP)
    # # corners2 = spread_out_corners(img2, VERTICAL_RECTANGLES, HORIZONTAL_RECTANGLES, HARRIS_CORNER_BOUNDRY_GAP)
    #
    # pyr1, filter1 = sol4_utils.build_gaussian_pyramid(img1, 3, 3)
    # pyr2, filter2 = sol4_utils.build_gaussian_pyramid(img2, 3, 3)
    # corners1, desc1 = find_features(pyr1)
    # corners2, desc2 = find_features(pyr2)
    # match_indices_1, match_indices_2 = match_features(desc1, desc2, MIN_SCORE)
    # # print(corners1[match_indices_1])
    # h_matrix, indices = ransac_homography(corners1[match_indices_1], corners2[match_indices_2], 550, 0.6)
    # # print(len(match_indices_1),len(match_indices_2))
    #
    # display_matches(img1, img2, corners1[match_indices_1], corners2[match_indices_2], indices)
    #
    # # plt.scatter(corners1[:, 0], corners1[:, 1])
    # # plt.show()


