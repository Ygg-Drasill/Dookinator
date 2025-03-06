import Homography as h
import numpy as np


def test_homography():
    cam = np.array([[145, 730], [1350, 1260], [990, 770], [1620, 960], [395, 840], [820, 1020]])
    absolute = np.array([[35.75, 20.16], [35.75, -20.16], [46.75, 9.16], [46.75, -9.16], [35.75, 7.31], [35.75, -7.31]])

    homo = h.homography(np.array([cam, absolute]))

    # test
    test = np.array([cam[0, 0], cam[0, 1], 1])
    homoGraphed = np.matmul(homo, test)
    homoGraphed = homoGraphed / homoGraphed[2]

    difference = np.array([absolute[0, 0], absolute[0, 1], 1]) - homoGraphed
    difference = difference.__abs__().max()

    # tolerance
    assert difference < 0.05


if __name__ == '__main__':
    homographyTest()
