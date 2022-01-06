import PIL.Image as pil_image
import PIL.ImageDraw as ImageDraw
import PIL.Image
import numpy as np
import os
from skimage import io


patch_size = 64
path = r'C:\\Users\\Hoven_Li\\Desktop\\B100\\B100_HR'
img_path = sorted([os.path.join(path, name) for name in os.listdir(path) if name.endswith('.png')])###这里的'.tif'可以换成任意的文件后缀
save_path = 'C:\\Users\\Hoven_Li\\Desktop\\B100'
LAM_img_path = 'C:\\Users\\Hoven_Li\\Documents\\GitHub\\LAM_Demo\\test_images\\7.png'
LAM_img = PIL.Image.open(LAM_img_path)
for i in range(len(img_path)):
    img_name = img_path[i]
    print(img_name)
    # img = PIL.Image.open(img_name)
    LAM_img_crop = LAM_img.crop((118-patch_size//2, 158-patch_size//2, 118+patch_size//2, 158+patch_size//2))
    LAM_img_crop.save(os.path.join(path, 'crop'+str(patch_size)+'.bmp'))
    # img.paste(LAM_img_crop, (118-patch_size//2, 158-patch_size//2, 118+patch_size//2, 158+patch_size//2))
    # img.save(img_name.replace('B100_HR', 'B100_p'+str(patch_size)))