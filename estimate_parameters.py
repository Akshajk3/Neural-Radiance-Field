import cv2
import numpy as np

from test import ray_directions

img1 = cv2.imread('novel_views/img_0.png')
img2 = cv2.imread('novel_views/img_1.png')

sift = cv2.SIFT.create()

keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = matcher.match(descriptors1, descriptors2)

matches = sorted(matches, key=lambda x: x.distance)

points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

E, mask = cv2.findEssentialMat(points1, points2, focal=0.1, pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=1.0)

inliers1 = points1[mask.ravel() == 1]
inliers2 = points2[mask.ravel() == 1]

_, R, t, _ = cv2.recoverPose(E, inliers1, inliers2)

P1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
P2 = np.hstack((R, t))

points1_hom = cv2.convertPointsToHomogeneous(inliers1).reshape(-1, 3)
points2_hom = cv2.convertPointsToHomogeneous(inliers2).reshape(-1, 3)

points_4d = cv2.triangulatePoints(P1, P2, points1_hom.T, points2_hom.T)
points3d = points_4d[:3] / points_4d[3]

points3d = points3d.T
points3d /= np.max(np.abs(points3d), axis=0)

ray_origins = []
ray_directions = []
pixel_values = []

for i, img in enumerate([img1, img2]):
    h, w = img.shape
    K = np.array()