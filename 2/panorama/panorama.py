import numpy as np
import cv2
import matplotlib.pyplot as plt

def detectAndDescribe(image):
    descriptor = cv2.SIFT()
    (kps, features) = descriptor.detectAndCompute(image, None)
    return (kps, features)

def calculateHomographyWithProvidedPoints():
    pan1_points = np.array([[57,64], [478,85],[76,367],[463,336]])
    pan2_points = np.array([[308, 56], [733, 48], [317, 314], [702, 347]])

    H, status = cv2.findHomography(pan1_points, pan2_points, 0)

    return H

def calculateHomographyWithSIFT(pan1, pan2):
    # Detection of keypoints
    pan1_kps, pan1_feat = detectAndDescribe(pan1)
    pan2_kps, pan2_feat = detectAndDescribe(pan2)

    # Feature matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(pan1_feat, pan2_feat, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    H, _ = cv2.findHomography(pan1_kps, pan2_kps, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
    return H


def main():
    # Loading images
    pan1 = cv2.imread("pan1.jpeg")
    pan2 = cv2.imread("pan2.jpeg")

    # Apply panorama correction
    width = pan1.shape[1] + pan2.shape[1]
    height = pan1.shape[0] + pan2.shape[0]

    # Finding the homography
    H = calculateHomographyWithProvidedPoints()

    result = cv2.warpPerspective(pan1, H, (width, height))
    result[0:pan2.shape[0], 0:pan2.shape[1]] = pan2

    plt.figure(figsize=(20,10))
    plt.imshow(result)

    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()