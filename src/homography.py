import numpy as np


def homography(pairs):
    source = pairs[0]
    dest = pairs[1]

    a = []

    length = len(source)
    i = 0

    while i < length:
        xs = source[i, 0]
        ys = source[i, 1]
        xd = dest[i, 0]
        yd = dest[i, 1]
        temp1 = np.array([-xs, -ys, -1, 0, 0, 0, xd * xs, xd * ys, xd])
        temp2 = np.array([0, 0, 0, -xs, -ys, -1, yd * xs, yd * ys, yd])
        a.append(temp1)
        a.append(temp2)
        i = i + 1

    a = np.asarray(a)

    u, s, vh = np.linalg.svd(a)
    h = vh[-1].reshape((3,3))
    h = h / h[2,2]

    return h
