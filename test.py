import glob
import os
import sys
import json
import numpy as np
import cv2
import scipy.io

# test code to display keypoints
data_path = '.\data'
mat_path = os.path.join(data_path, 'labels\B1Counting_BB.mat')
B1_path = os.path.join(data_path, 'images\B1Counting')
B1_mat = scipy.io.loadmat(mat_path)


img_index = 500
img_name = f"BB_left_{img_index}.png"
img = cv2.imread(os.path.join(B1_path, img_name), 1)
print(img_name)

B1_keypoints = B1_mat['handPara'][:, :, img_index]
BB_Kmat = np.array([[822.79041, 0, 318.47345],[0, 822.79041, 250.31296],[0,0,1]])
transformed_keypoints = BB_Kmat.dot(B1_keypoints)
for i in range(transformed_keypoints.shape[1]):
    normalization_factor = transformed_keypoints[2, i]
    normalized_keypoint = (transformed_keypoints[:2, i] / normalization_factor).astype(int)
    xy = (normalized_keypoint[0], normalized_keypoint[1])
    cv2.circle(img, xy, 3, (255, 0, 0), -1)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()