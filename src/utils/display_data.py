import sys
import os
import re
import numpy as np
import cv2
import scipy.io
from utils.matrices import P_matrix, apply_homogeneous_transform, reduce_homogeneous

def _circle_uv_points(img, uvs):
	# uv should be (2, N) array
	
	for i in range(uvs.shape[1]):
		keypoint = uvs[:, i]
		uv = (keypoint[0], keypoint[1])
		img = cv2.circle(img, uv, 3, (255, 0, 0), -1)
	return img

def main():
	# sys.argv[1] should be something like 'B1Counting/BB_left_0.png'
	head, img_name = os.path.split(sys.argv[1])
	img_index = int(re.search(r'\d+', img_name).group(0))
	data_path = './data'
	left_image_path = os.path.join(data_path, 'images/' + sys.argv[1])
	right_image_path = left_image_path.replace('left', 'right')
	mat_name = head + '_' + img_name[:2] + '.mat' 
	mat_path = os.path.join(data_path, 'labels/' + mat_name)
	
	left_img = cv2.imread(left_image_path, 1)
	right_img = cv2.imread(right_image_path, 1)
	mat = scipy.io.loadmat(mat_path)
	keypoints = mat['handPara'][:,:,img_index]

	P = P_matrix(822.79041, 822.79041, 318.47345, 250.31296, 120.054)
	uvds = reduce_homogeneous(apply_homogeneous_transform(keypoints, P))
	left_uvs = uvds[:2].astype(int)
	uvds[0] -= uvds[2]
	right_uvs = uvds[:2].astype(int)

	left_img_with_keypoints = _circle_uv_points(left_img, left_uvs)
	right_img_with_keypoints = _circle_uv_points(right_img, right_uvs)

	cv2.imshow('Left Image', left_img_with_keypoints)
	cv2.imshow('Right Image', right_img_with_keypoints)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	assert len(sys.argv) == 2, "must contain single path argument"
	main()