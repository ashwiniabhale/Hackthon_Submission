from PIL import Image

from resizeimage import resizeimage


with open('test4.png', 'r+b') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [494, 322])
        cover.save('test4modify.png', image.format)