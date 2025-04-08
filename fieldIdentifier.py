import cv2
import cv2 as cv
import datetime
import numpy as np


def fieldIdentifier(frame):
    time = datetime.datetime.now()
    video = cv.VideoCapture("Soccer.mp4")
    success, source = video.read()
    #change 749 to frame
    video.set(cv2.CAP_PROP_POS_FRAMES, 749)
    success2, source2 = video.read()

    #source2 = cv.imread("ClearSoccer.png")

    source = source[130:,:]
    source2 = source2[130:,:]
    #source = cv.blur(source,[5,5])
    #source2 = cv.blur(source2, [5,5])

    red = source.copy()
    red2 = source2.copy()

    red[:, :, 0] = 0
    red[:, :, 1] = 0
    red2[:, :, 0] = 0
    red2[:, :, 0] = 0

    sift = cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(red,None)
    kp2, des2 = sift.detectAndCompute(red2, None)

    FLANN_INDEX_KDTREE = 1
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
    searchParams = dict(checks=50)
    flann = cv.FlannBasedMatcher(indexParams,searchParams)
    nNeighbors = 2
    matches = flann.knnMatch(des1,des2,k=nNeighbors)

    print(len(matches))

    goodMatches = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            goodMatches.append(m)

    minGoodMatches = 10

    if len(goodMatches) >= minGoodMatches:
        srcPts = np.float32([kp1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        errorThreshold = 5
        M,mask = cv.findHomography(srcPts,dstPts,cv.RANSAC,errorThreshold)
        matchesMask = mask.ravel().tolist()
        shape = source.shape
        h = shape[0]
        w = shape[1]
        imgBorder = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        warpedImgBorder = cv.perspectiveTransform(imgBorder,M)
        source2 = cv.polylines(source2,[np.int32(warpedImgBorder)],True,255,3,cv.LINE_AA)
    else:
        print("Not enough matches")
        matchesMask = None

    green = [0,255,0]
    drawParams = dict(matchColor=green,singlePointColor=None,matchesMask=matchesMask,flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    matchImg = cv.drawMatches(source,kp1,source2,kp2,goodMatches,None,**drawParams)

    time = datetime.datetime.now() - time

    print(time)

    cv.imshow("matches", matchImg)
    cv.waitKey()
    cv.imwrite("matches.png", matchImg)
