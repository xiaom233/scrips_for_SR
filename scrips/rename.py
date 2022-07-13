
import numpy as np
import os
from skimage import io

path = r'E:\\dataset\\benchmark_ready\\\Manga109\\LRbicx4'
img_path = sorted([os.path.join(path, name) for name in os.listdir(path) if name.endswith('.png')])###这里的'.tif'可以换成任意的文件后缀
for i in range(len(img_path)):
    img_name = img_path[i]
    print(img_name)
    name = img_name.split('_LR')[0]
    ext = img_name.split('_LR')[1]
    print(name+ext)
    os.rename(img_name, name+ext)
