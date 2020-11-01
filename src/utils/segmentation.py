import os
import sys
import glob
import numpy as np
import cv2

def normalizedRGB(rgb_image):
    # should have shape (R, C, 3)
    total = rgb_image.sum(axis=2)
    return rgb_image / total[:,:,np.newaxis]

def skin_color_mask_from_RGB(normalized_rgb):
    r, g, b = normalized_rgb.transpose((2, 0, 1))
    r_over_g = r / g
    r_mul_b = r * b
    r_mul_g = r * g
    mask_1 = (r_over_g > 1.185)
    # mask_2 = (r_mul_b > 0.107)
    # mask_3 = (r_mul_g > 0.112)
    return mask_1

def skin_color_mask_from_HSV(hsv):
    h, s, v = hsv.transpose((2,0,1))
    h = h * 2
    s = s / 255
    # v = v / 255
    h_mask = ((h > 0) & (h < 25)) | ((h > 335) & (h < 360))
    s_mask = (s > 0.2) & (s < 0.6)
    # v_mask = (v >= 0.4)
    return h_mask & s_mask

def skin_color_mask_from_YCRCB(ycrcb):
    y, cr, cb = ycrcb.transpose((2,0,1))
    cr_mask = (cr >= 133) & (cr <= 173)
    cb_mask = (cb >= 77) & (cb <= 127)
    return cr_mask & cb_mask

def skin_color_mask(img):
    # input should be BGR image
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    rgb_mask = skin_color_mask_from_RGB(normalizedRGB(rgb_img))
    hsv_mask = skin_color_mask_from_HSV(hsv_img)
    ycrcb_mask = skin_color_mask_from_YCRCB(ycrcb_img)

    return rgb_mask & hsv_mask & ycrcb_mask

def apply_skin_color_mask(img):
    # input should be BGR image
    mask = skin_color_mask(img)
    return mask.astype(np.uint8)[:,:,np.newaxis] * img

def main():
    data_path = '../../data'
    B1_path = os.path.join(data_path, 'images/B1Counting')
    img = cv2.imread(os.path.join(B1_path, sys.argv[1]), 1)

    # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # rgb_norm_img = normalizedRGB(rgb_img)
    # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    img_mask = skin_color_mask(img)
    applied_mask = apply_skin_color_mask(img)

    cv2.imshow('RGB', img)
    cv2.imshow('Skin Color Mask', img_mask.astype(float))
    cv2.imshow('Applied Color Mask', applied_mask)
    # cv2.imshow('RGB Mask', rgb_mask.astype(float))
    # cv2.imshow('HSV Mask', hsv_mask.astype(float))
    # cv2.imshow('YCrCb Mask', ycrcb_mask.astype(float))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    assert len(sys.argv) == 2, "should have exactly one argument; name of png"
    main()






