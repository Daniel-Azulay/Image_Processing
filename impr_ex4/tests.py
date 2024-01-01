import unittest

import imageio
import numpy as np
import scipy
from scipy.stats import rankdata

import sol4
import matplotlib.pyplot as plt
import sol4_utils

class TestSol4(unittest.TestCase):
    def test_harris_corner_detector(self):
        im = sol4_utils.read_image("external/blox.jpg", 1)
        coord = sol4.spread_out_corners(im, 7, 7, 7)
        # coord = sol4.harris_corner_detector(im)
        plt.figure(figsize=(5,5))
        plt.imshow(im, cmap='gray')
        plt.scatter(coord[:, 0], coord[:, 1], marker='.', c='r')
        plt.show()

    # def test(self):
    #     im1 = sol4_utils.read_image("external/oxford1.jpg", 1)[:,400:-1]
    #     im2 = sol4_utils.read_image("external/oxford2.jpg", 1)[:,0:400]
    #     pyr1,_ = sol4_utils.build_gaussian_pyramid(im1, 3, 7)
    #     pyr2,_ = sol4_utils.build_gaussian_pyramid(im2, 3, 7)
    #
    #     c1, d1 = sol4.find_features(pyr1)
    #     c2, d2 = sol4.find_features(pyr2)
    #     m = sol4.match_features(d1, d2, 0.5)
    #
    #     fig, (ax_orig, ax_mag) = plt.subplots(1, 2, figsize=(20, 10))
    #     ax_orig.imshow(im1, cmap='gray')
    #     ax_orig.set_title('im1')
    #     ax_orig.scatter(c1[:, 0], c1[:, 1], marker='.', c='b')
    #     ax_orig.scatter(c1[m[0], 0], c1[m[0], 1], marker='.', c='r')
    #     ax_orig.set_axis_off()
    #
    #     ax_mag.imshow(im2, cmap='gray')
    #     ax_mag.set_title('im2')
    #     ax_mag.scatter(c2[:, 0], c2[:, 1], marker='.', c='b')
    #     ax_mag.scatter(c2[m[1], 0], c2[m[1], 1], marker='.', c='r')
    #     ax_mag.set_axis_off()
    #     fig.show()

    def test_match_features(self):
        im1 = sol4_utils.read_image("external/oxford1.jpg", 1)
        im2 = sol4_utils.read_image("external/oxford2.jpg", 1)
        im = np.hstack([im1, im2])
        width = im1.shape[1]
        pyr1, _ = sol4_utils.build_gaussian_pyramid(im1, 3, 5)
        pyr2, _ = sol4_utils.build_gaussian_pyramid(im2, 3, 5)

        c1, d1 = sol4.find_features(pyr1)
        c2, d2 = sol4.find_features(pyr2)
        m = sol4.match_features(d1, d2, 0.99)
        c2[:, 0] = c2[:, 0] + width
        plt.figure(figsize=(16, 16))
        plt.imshow(im, cmap='gray')

        plt.scatter(c1[:, 0], c1[:, 1], marker='.', c='b')
        plt.scatter(c1[m[0], 0], c1[m[0], 1], marker='.', c='r')
        plt.scatter(c2[:, 0], c2[:, 1], marker='.', c='b')
        plt.scatter(c2[m[1], 0], c2[m[1], 1], marker='.', c='r')
        for i in range(m[0].size):
            plt.plot([c1[m[0][i], 0], c2[m[1][i], 0]],
                     [c1[m[0][i], 1], c2[m[1][i], 1]], mfc='r', c ='y', lw=.4, ms=10, marker='o')
        plt.show()

    def test_ransac(self):
        im1 = sol4_utils.read_image("external/oxford1.jpg", 1)
        im2 = sol4_utils.read_image("external/oxford2.jpg", 1)

        # im1 = sol4_utils.read_image("dump/peru_rev/peru_rev486.jpg", 1)
        # im2 = sol4_utils.read_image("dump/peru_rev/peru_rev487.jpg", 1)

        pyr1, _ = sol4_utils.build_gaussian_pyramid(im1, 3, 3)
        pyr2, _ = sol4_utils.build_gaussian_pyramid(im2, 3, 3)

        c1, d1 = sol4.find_features(pyr1)
        c2, d2 = sol4.find_features(pyr2)
        m = sol4.match_features(d1, d2, 0.9)
        m1, m2 = c1[m[0]], c2[m[1]]

        homo, besti = sol4.ransac_homography(m1, m2, 100, 5)
        sol4.display_matches(im1, im2, m1, m2, besti)


    def test_accumulate_homographies(self):
        from numpy.linalg import inv
        H01 = np.random.choice(np.linspace(0, 10, 1000), size=9).reshape(3, 3)
        H12 = np.random.choice(np.linspace(0, 10, 1000), size=9).reshape(3, 3)
        H23 = np.random.choice(np.linspace(0, 10, 1000), size=9).reshape(3, 3)
        H34 = np.random.choice(np.linspace(0, 10, 1000), size=9).reshape(3, 3)
        H45 = np.random.choice(np.linspace(0, 10, 1000), size=9).reshape(3, 3)

        m=3
        expected = [H23 @ H12 @ H01,
                    H23 @ H12,
                    H23,
                    np.eye(3),
                    inv(H34),
                    inv(H34) @ inv(H45)]
        got = sol4.accumulate_homographies([H01, H12, H23, H34, H45], m)

        expected = [np.eye(3), inv(H01)]
        got = sol4.accumulate_homographies([H01], 0)
        return

    def apply_homography_old(pos1, H12):
        """
        Apply homography to inhomogenous points.
        :param pos1: An array with shape (N,2) of [x,y] point coordinates.
        :param H12: A 3x3 homography matrix.
        :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
        """
        x, y = pos1[0], pos1[1]
        new_x = x * H12[0, 0] + y * H12[0, 1] + H12[0, 2]
        new_y = x * H12[1, 0] + y * H12[1, 1] + H12[1, 2]
        factor = x * H12[2, 0] + y * H12[2, 1] + H12[2, 2]
        return np.array([new_x / factor, new_y / factor])

    def test_warping(self):
        im1 = sol4_utils.read_image("external/oxford1.jpg", 1)
        im2 = sol4_utils.read_image("external/oxford2.jpg", 1)
        h1, w1 = im1.shape
        h2, w2 = im2.shape

        pyr1, _ = sol4_utils.build_gaussian_pyramid(im1, 3, 5)
        pyr2, _ = sol4_utils.build_gaussian_pyramid(im2, 3, 5)
        c1, d1 = sol4.find_features(pyr1)
        c2, d2 = sol4.find_features(pyr2)
        m = sol4.match_features(d1, d2, 0.7)

        m1, m2 = c1[m[0]], c2[m[1]]
        homo, besti = sol4.ransac_homography(m1, m2, 100, 50)
        sol4.display_matches(im1, im2, m1, m2, besti)

        b1 = sol4.compute_bounding_box(np.eye(3), w1, h1)
        b2 = sol4.compute_bounding_box(np.linalg.inv(homo), w2, h2)

        xcenter1 = w1 / 2
        xcenter2 = sol4.apply_homography(np.array([[w2/2, h2/2]]), np.linalg.inv(homo))[0, 0]
        seam = (xcenter1 + xcenter2) // 2

        im2_translated = sol4.warp_channel(im2, np.linalg.inv(homo))
        # plt.figure()
        # plt.imshow(im2_translated, cmap="gray")
        # plt.show()

        im1 = np.pad(im1, ((max(0, -b2[0, 1]), 0), (0, 0)))
        diff = im1.shape[0]-im2_translated.shape[0]
        if diff > 0:
            im2_translated = np.pad(im2_translated, ((-min(0, -b2[0, 1]), diff), (0, 0)))
        else:
            im1 = np.pad(im1, ((0, -diff), (0, 0)))
        im = np.hstack([im1[:, :int(seam)], im2_translated[:, int(seam-b2[0, 0]):]])
        imageio.imwrite("pano.jpg", im)
        plt.figure()
        plt.imshow(im, cmap="gray")
        plt.show()

        return

if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()