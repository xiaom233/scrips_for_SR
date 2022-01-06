from collections import OrderedDict
import matplotlib.pyplot as plt
import cv2 as cv
import os

import numpy as np


def sobel_demo(image):
    grad_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    gradx = cv.convertScaleAbs(grad_x)
    grad_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
    grady = cv.convertScaleAbs(grad_y)
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.imshow('gradxy', gradxy)
    cv.waitKey(0)
    img_np = cv.mean(gradxy)
    mean = (img_np[0] + img_np[1] + img_np[2])/3.
    # print(img_np)
    return mean


if __name__ == '__main__':
    # dir_path = 'E:\\dataset\\visual_results_001_bicSR_DIV2K_s48w8_SwinIR-M_x4\\001_bicSR_DIV2K_s48w8_SwinIR-M_x4\\visualization\\B100'
    dir_path = 'C:\\Users\\Hoven_Li\\Documents\\GitHub\\BasicSR_xiaom\\datasets\\Manga109\\original'
    # dir_path = 'C:\\Users\\Hoven_Li\\Documents\\GitHub\\BasicSR_xiaom\\datasets\\DIV2K/DIV2K_train_HR_sub'

    img_path = sorted([os.path.join(dir_path, name) for name in os.listdir(dir_path) if
                       (name.endswith('.png'))])  ###这里的'.tif'可以换成任意的文件后缀
    mean_dic = OrderedDict()
    mean_dic['mean'] = []
    for i in range(len(img_path)):
        print(img_path[i])
        image = cv.imread(img_path[i])
        mean = sobel_demo(image)
        mean_dic['mean'].append(mean)
    mean_array = np.array(mean_dic['mean'])
    ave_mean = sum(mean_dic['mean']) / len(mean_dic['mean'])
    print(ave_mean)
    # plt.hist(mean_array, histtype='step', bins=50, alpha=0.5)
    # plt.show()
