import PIL.Image as pil_image
import PIL.ImageDraw as ImageDraw
import PIL.Image

input_path =    'C:\\Users\\Hoven_Li\\Desktop\\visual\\U100_img_076\\img076x4_SwinIR.png'
GT_path =       'C:\\Users\\Hoven_Li\\Desktop\\visual\\U100_img_076\\img_004_GT.png'
GT_output =     'C:\\Users\\Hoven_Li\\Desktop\\visual\\U100_img_076\\img_004_GT_rec.png'
GT_Crop_path =  'C:\\Users\\Hoven_Li\\Desktop\\visual\\U100_img_076\\result\\img_004_GT_crop1.png'
output_path_1 = 'C:\\Users\\Hoven_Li\\Desktop\\visual\\U100_img_076\\result\\img076x4_SwinIR_result1.png'
# output_path_2 = 'C:\\Users\\Hoven_Li\\Desktop\\visual\\U100_img_073\\result\\img004x4_SwinIR_crop2.png'

# output crop
# Manga109_YumeiroCooking(25, 800, 125, 850)
# U100_img_004(515, 280, 635, 400)(660, 480, 760, 580)
# U100_img_073(60, 5, 140, 35)
# U100_img_076(275, 215, 385, 250)
output = PIL.Image.open(input_path)
crop_1 = output.crop((275, 215, 385, 250))
# crop_2 = output.crop((660, 480, 760, 580))

# GT_crop
# GT = PIL.Image.open(GT_path)
# GT_crop = GT.crop((25, 800, 125, 850))

# GT rectangled
# draw_1 = ImageDraw.Draw(GT)
# draw = draw_1.rectangle((25, 800, 125, 850), outline="red", width=3)
# draw_2 = ImageDraw.Draw(output)
# draw_2 = draw_2.rectangle((515, 280, 635, 400), outline="red", width=3)

# crop_height = 150
# crop = crop.resize((crop_height, crop_height), resample=PIL.Image.NEAREST)
# height, width = output.size
# output.paste(crop, (height-crop_height, width-crop_height, height, width))
try:
    crop_1.save(output_path_1)
    # crop_2.save(output_path_2)
    # GT_crop.save(GT_Crop_path)
    # GT.save(GT_output)
except AttributeError:
    print("Couldn't save image {}".format(output_path_1))