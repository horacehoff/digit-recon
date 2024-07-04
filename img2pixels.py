from PIL import Image


def img_to_pixels(path):
    im = Image.open(path).convert("RGB")
    px = im.load()
    width, height = im.size
    pixels = []
    for x in range(height):
        for y in range(width):
            # BACKGROUND == 0
            # DIGIT'S SHAPE == 1
            if px[y, x][0] == 255 and px[y, x][1] == 255 and px[y, x][2] == 255:
                pixels.append(0)
            else:
                pixels.append(1)
    return pixels