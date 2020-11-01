import sys
import os
import re
import numpy as np
import cv2
import scipy.io
from matrices import P_matrix, apply_homogeneous_transform, reduce_homogeneous

def circle_keypoints(img, keypoints):
	# currently hardcoded; gonna change later
	P = P_matrix(822.79041, 822.79041, 318.47345, 250.31296, 120.054)
	uvds = reduce_homogeneous(apply_homogeneous_transform(keypoints, P)).astype(int)
	for i in range(uvds.shape[1]):
		keypoint = uvds[:, i]
		uv = (keypoint[0], keypoint[1])
		img = cv2.circle(img, uv, 3, (255, 0, 0), -1)
	return img

def main():
	# sys.argv[1] should be something like 'B1Counting/BB_left_0.png'
	head, img_name = os.path.split(sys.argv[1])
	img_index = int(re.search(r'\d+', img_name).group(0))
	data_path = '../../data'
	image_path = os.path.join(data_path, 'images/' + sys.argv[1])
	mat_name = head + '_' + img_name[:2] + '.mat' 
	mat_path = os.path.join(data_path, 'labels/' + mat_name)
	
	img = cv2.imread(image_path, 1)
	mat = scipy.io.loadmat(mat_path)
	keypoints = mat['handPara'][:,:,img_index]

	img_with_keypoints = circle_keypoints(img, keypoints)

	cv2.imshow('image',img_with_keypoints)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	assert len(sys.argv) == 2, "must contain single path argument"
	main()