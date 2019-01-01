import numpy as np
import cv2
import random

UBIT = 'nikhilsr'
np.random.seed(sum([ord(c) for c in UBIT]))

mountain1ImageLocation = './proj2_data/data/mountain1.jpg'
mountain2ImageLocation = './proj2_data/data/mountain2.jpg'


def writeImage(img, outputFileName):
    cv2.imwrite(outputFileName, img)
    return 1


def readImage(imageLocation):
    img = cv2.imread(imageLocation, 1)
    return img


def main():
    print("Task 1 :")
    print(" Task 1.1 : ")
    img1 = readImage(mountain1ImageLocation)
    img2 = readImage(mountain2ImageLocation)
    img1_ = img1.copy()
    img2_ = img2.copy()
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    img1_keyPoints, img1_desc = sift.detectAndCompute(img1_gray, None)
    img2_keyPoints, img2_desc = sift.detectAndCompute(img2_gray, None)
    img1_withHighlightedKeyPoints = cv2.drawKeypoints(img1_gray, img1_keyPoints, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_withHighlightedKeyPoints = cv2.drawKeypoints(img2_gray, img2_keyPoints, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    writeImage(img1_withHighlightedKeyPoints, "task1_sift1.jpg")
    writeImage(img2_withHighlightedKeyPoints, "task1_sift2.jpg")
    print("     Task 1.1 Completed. Keypoints have been detected.")
    print(" Task 1.2 : ")
    K = 2
    M2NRatio = 0.75
    FLANN_INDEX_KDTREE = 0
    numOfChecks = 100
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = numOfChecks)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(img1_desc, img2_desc, k=K)
    goodMatches = []
    for m, n in matches:
        if (m.distance < (M2NRatio * n.distance)):
            goodMatches.append(m)
    knnMatchedImg = cv2.drawMatches(img1_withHighlightedKeyPoints, img1_keyPoints, img2_withHighlightedKeyPoints, img2_keyPoints, goodMatches, None, flags=2)     
    writeImage(knnMatchedImg, "task1_matches_knn.jpg")
    print("     Task 1.2 Completed. Good Keypoint matches have been mapped.")
    print(" Task 1.3 : ")
    H = None
    MIN_MATCH_COUNT=10
    if (len(goodMatches) >= MIN_MATCH_COUNT):
        src_pts = np.float32([ img1_keyPoints[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
        dst_pts = np.float32([ img2_keyPoints[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        print("     The Homography matrix H : ")
        print("     "+str(H))
    else:
        print("     Not enough good matches were found with minimum match count being set at "+str(MIN_MATCH_COUNT))
    print("     Task 1.3 Completed.")
    print(" Task 1.4 : ")
    MATCHES_TO_BE_MAPPED = 10
    MIN_MATCH_COUNT = 10
    goodMatches1 = random.sample(goodMatches, MATCHES_TO_BE_MAPPED)
    if (len(goodMatches1) >= MIN_MATCH_COUNT):
        src_pts1 = np.float32([ img1_keyPoints[m.queryIdx].pt for m in goodMatches1 ]).reshape(-1,1,2)
        dst_pts1 = np.float32([ img2_keyPoints[m.trainIdx].pt for m in goodMatches1 ]).reshape(-1,1,2)
        H1, mask1 = cv2.findHomography(src_pts1, dst_pts1, cv2.RANSAC, 5.0)
        matchesMask1 = mask1.ravel().tolist()
        height1, width1, channels1 = img1_withHighlightedKeyPoints.shape
        pts1 = np.float32([ [0,0],[0,height1-1],[width1-1,height1-1],[width1-1,0] ]).reshape(-1,1,2)
        dst1 = cv2.perspectiveTransform(pts1,H1)
    else:
        print("     Not enough good matches were found with minimum match count being set at "+str(MIN_MATCH_COUNT))
        matchesMask1 = None
    draw_params = dict( matchColor = (0,255,255), singlePointColor = None, matchesMask = matchesMask1, flags = 2 )
    matchedImg = cv2.drawMatches(img1_withHighlightedKeyPoints, img1_keyPoints, img2_withHighlightedKeyPoints, img2_keyPoints, goodMatches1, None, **draw_params)
    writeImage(matchedImg, "task1_matches.jpg")
    print("     Task 1.4 Completed.")
    print(" Task 1.5 : ")
    img1_height, img1_width = img1_.shape[:2]
    img1_pixels = np.float32([ [0,0],[0,img1_height],[img1_width,img1_height],[img1_width,0] ]).reshape(-1,1,2)
    img2_height, img2_width = img2_.shape[:2]
    img2_pixels = np.float32([ [0,0],[0,img2_height],[img2_width,img2_height],[img2_width,0] ]).reshape(-1,1,2)
    img2_pixels = cv2.perspectiveTransform(img2_pixels, H)
    warpedImg_pixels = np.concatenate((img1_pixels, img2_pixels), axis=0)
    [xMin, yMin] = np.int32( warpedImg_pixels.min(axis=0).ravel() - 0.5 )
    [xMax, yMax] = np.int32( warpedImg_pixels.max(axis=0).ravel() + 0.5 )
    t = [-xMin, -yMin]
    Ht = np.array([ [1,0,t[0]],[0,1,t[1]],[0,0,1] ])
    warpedImg = cv2.warpPerspective(img2_, Ht.dot(H), (xMax-xMin, yMax-yMin))
    warpedImg[ t[1]:(img1_height+t[1]), t[0]:(img1_width+t[0]) ] = img1_
    writeImage(warpedImg, "task1_pano.jpg")
    print("     Task 1.5 Completed.")
    

main()