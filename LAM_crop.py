import os
import PIL.Image
import PIL.Image as pil_image

patch_size = 16
path = r'C:\\Users\\Hoven_Li\\Desktop\\visual\\\LAM\\\result_7'
img_path = sorted([os.path.join(path, name) for name in os.listdir(path) if (name.endswith('.bmp') and ('Base' in name))])###这里的'.tif'可以换成任意的文件后缀
for i in range(len(img_path)):
    img_name = img_path[i]
    print(img_name)
    img = PIL.Image.open(img_name)
    crop = img.crop((1140, 150, 1140 + patch_size, 150 + patch_size))
    crop.save(img_name.replace('@Base', '_crop16'))