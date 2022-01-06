import cv2
import os

GT_path = 'C:\\Users\\Hoven_Li\\Desktop\\visual\\residual_imges_Urban100\\GTmod4'
IR_path = 'C:\\Users\\Hoven_Li\\Desktop\\visual\\residual_imges_Urban100\\SwinIR_Urban100'
model_name = 'SwinIR'
img_path = sorted([os.path.join(IR_path, name) for name in os.listdir(IR_path) if (name.endswith('.png'))])###这里的'.tif'可以换成任意的文件后缀
for i in range(len(img_path)):
    print(img_path[i])
    hr = cv2.imread(img_path[i])
    if model_name == 'SwinIR':
        gt = cv2.imread(img_path[i].replace('x4_SwinIR', '').replace('SwinIR_Urban100', 'GTmod4').replace('g0', 'g_0'))
        print(img_path[i].replace('x4_SwinIR', '').replace('SwinIR_Urban100', 'GTmod4').replace('g0', 'g_0'))
    print(gt.shape)
    print(hr.shape)
    residual = cv2.absdiff(gt, hr)
    cv2.imwrite(img_path[i].replace('SwinIR_Urban100', 'SwinIR_residual'), residual)