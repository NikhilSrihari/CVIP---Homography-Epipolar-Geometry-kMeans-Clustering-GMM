import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib import cm
import math

UBIT = 'nikhilsr'
np.random.seed(sum([ord(c) for c in UBIT]))

tsucubaLeft_ImageLocation = './proj2_data/data/tsucuba_left.png'
tsucubaRight_ImageLocation = './proj2_data/data/tsucuba_right.png'


def writeImage(img, outputFileName):
    cv2.imwrite(outputFileName, img)
    return 1


def readImage(imageLocation):
    img = cv2.imread(imageLocation, 1)
    return img


def main():
    print("Task 2 :")
    print(" Task 2.1 : ")
    img1 = readImage(tsucubaLeft_ImageLocation)
    img2 = readImage(tsucubaRight_ImageLocation)
    img1_ = img1.copy()
    img2_ = img2.copy()
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    img1_keyPoints, img1_desc = sift.detectAndCompute(img1_gray, None)
    img2_keyPoints, img2_desc = sift.detectAndCompute(img2_gray, None)
    img1_withHighlightedKeyPoints = cv2.drawKeypoints(img1_gray, img1_keyPoints, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_withHighlightedKeyPoints = cv2.drawKeypoints(img2_gray, img2_keyPoints, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    writeImage(img1_withHighlightedKeyPoints, "task2_sift1.jpg")
    writeImage(img2_withHighlightedKeyPoints, "task2_sift2.jpg")
    K = 2
    M2NRatio = 0.75
    FLANN_INDEX_KDTREE = 1
    numOfChecks = 100
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = numOfChecks)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(img1_desc, img2_desc, k=K)
    goodMatches = []
    img1_points = []
    img2_points = []
    for m, n in matches:
        if (m.distance < (M2NRatio * n.distance)):
            goodMatches.append(m)
            img1_points.append(img1_keyPoints[m.queryIdx].pt)
            img2_points.append(img2_keyPoints[m.trainIdx].pt)
    knnMatchedImg = cv2.drawMatches(img1_withHighlightedKeyPoints, img1_keyPoints, img2_withHighlightedKeyPoints, img2_keyPoints, goodMatches, None, flags=2)     
    writeImage(knnMatchedImg, "task2_matches_knn.jpg")
    print("     Task 2.1 Completed. Keypoints have been detected. KNN matched image has been created")
    print(" Task 2.2 : ")
    #img1_points = np.array(random.sample(list(np.int32(img1_points)), 10))
    #img2_points = np.array(random.sample(list(np.int32(img2_points)), 10))
    img1_points = np.array(random.sample(img1_points, 10), dtype=np.int32)
    img2_points = np.array(random.sample(img2_points, 10), dtype=np.int32)
    F, mask = cv2.findFundamentalMat(img1_points, img2_points, cv2.FM_LMEDS)
    print("     The Fundamental Matrix F : ")
    print("     "+str(F))
    print("     Task 2.2 Completed.")
    print(" Task 2.3 : ")
    img1_inlierPoints = img1_points[mask.ravel()==1]
    img2_inlierPoints = img2_points[mask.ravel()==1]
    el_LonR = (cv2.computeCorrespondEpilines(img1_inlierPoints.reshape(-1,1,2), 1, F)).reshape(-1,3)
    el_RonL = (cv2.computeCorrespondEpilines(img2_inlierPoints.reshape(-1,1,2), 2, F)).reshape(-1,3)
    r,c,v = img1.shape
    for r, img1_inlierPoint, img2_inlierPoint in zip(el_RonL, img1_inlierPoints, img2_inlierPoints):
        color = (0,255,255)
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img_epi_left = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img_epi_left = cv2.circle(img_epi_left, tuple(img1_inlierPoint), 5, color, -1)
    r,c,v = img2.shape
    for r, img2_inlierPoint, img1_inlierPoint in zip(el_LonR, img2_inlierPoints, img1_inlierPoints):
        color = (255,255,0)
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img_epi_right = cv2.line(img2, (x0,y0), (x1,y1), color, 1)
        img_epi_right = cv2.circle(img_epi_right, tuple(img2_inlierPoint), 5, color, -1)
    writeImage(img_epi_left, "task2_epi_left.jpg")
    writeImage(img_epi_right, "task2_epi_right.jpg")
    print("     Task 2.3 Completed.")
    print(" Task 2.4 : ")
    stereo = cv2.StereoSGBM_create(numDisparities=64, blockSize=25)
    img_disparity = stereo.compute(img1_gray, img2_gray)
    thresholdImg = (cv2.threshold(img_disparity, 0.6, 1.0, cv2.THRESH_BINARY))[1]
    #writeImage(img_disparity, "task2_disparity.jpg")
    plt.subplot(122)
    plt.imsave('task2_disparity.jpg', img_disparity, cmap = cm.gray)
    print("     Task 2.4 Completed.")


main()