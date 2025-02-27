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


if __name__ == '__main__':
    #outer goal right, outer goal left, inner goal right, inner goal left, goal circle right, goal circle left
    # field length 105,68   top right from centre 52.25,34   outer goal from centre 52.25,20.16
    #0,630   660,585   2304,1023   2304,1728   0,1728
    cam = np.array([[145,730],[1350,1260],[990,770],[1620,960],[395,840],[820,1020]])
    absolute = np.array([[35.75,20.16],[35.75,-20.16],[46.75,9.16],[46.75,-9.16],[35.75,7.31],[35.75,-7.31]])

    homo = homography(np.array([cam,absolute]))
    print(homo)

    #test
    test = np.array([cam[0,0], cam[0,1], 1])
    homoGraphed = np.matmul(homo, test)
    homoGraphed = homoGraphed / homoGraphed[2]
    print("got: ")
    print(homoGraphed)
    print("expected: ")
    print(absolute[0])
    print("from: ")
    print(cam[0])
