from typing import List
from PIL import Image

import numpy as np


node_colors = [
    "#cc5151","#7f3333","#51cccc","#337f7f","#8ecc51","#597f33","#8e51cc","#59337f","#ccad51","#7f6c33",
    "#51cc70","#337f46","#5170cc","#33467f","#cc51ad","#7f336c","#cc7f51","#7f4f33","#bccc51","#757f33",
    "#60cc51","#3c7f33","#51cc9e","#337f62","#519ecc","#33627f","#6051cc","#3c337f","#bc51cc","#75337f",
    "#cc517f","#7f334f","#cc6851","#7f4133","#cc9651","#7f5e33","#ccc451","#7f7a33","#a5cc51","#677f33",
    "#77cc51","#4a7f33","#51cc59","#337f37","#51cc87","#337f54","#51ccb5","#337f71","#51b5cc","#33717f",
    "#5187cc","#33547f","#5159cc","#33377f","#7751cc","#4a337f","#a551cc","#67337f","#cc51c4","#7f337a",
    "#cc5196","#7f335e","#cc5168","#7f3341","#cc5d51","#7f3a33","#cc7451","#7f4833","#cc8a51","#7f5633",
    "#cca151","#7f6533","#ccb851","#7f7333","#c8cc51","#7d7f33","#b1cc51","#6e7f33","#9acc51","#607f33",
    "#83cc51","#527f33","#6ccc51","#437f33","#55cc51","#357f33","#51cc64","#337f3e","#51cc7b","#337f4d",
    "#51cc92","#337f5b","#51cca9","#337f69","#51ccc0","#337f78","#51c0cc","#33787f","#51a9cc","#33697f"
]
num_colors = len(node_colors)


def get_node_color(i):
    """
    Returns a color from the node_colors list.
    """
    return node_colors[i % num_colors]


def feat_to_alpha_value(feat):
    """
    Converts a feature value to an alpha value in the range [0, 255].
    """
    return max(min(int(feat*255), 255), 0)


def hex_to_rgb_tuple(c):
    """
    Converts a hex color code to an RGB tuple.
    """
    h = c.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def hex_to_rgba_tuple(c):
    """
    Converts a hex color code to an RGBA tuple.
    """
    h = c.lstrip('#') + 'ff'
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4, 6))


def angle_to_x_axis(y, x):
    """
    Returns the angle between the vector (y, x) and the x-axis in degrees.
    """
    angle_radians = np.arctan2(y, x)
    angle_degrees = 180 * angle_radians / np.pi
    if angle_degrees < 0:
        angle_degrees += 360.0
    return angle_degrees


def pad_images_to_largest(imgs: List[Image.Image]) -> List[Image.Image]:
    """
    Pad all images in the iterable to the largest image (width and height considered separately).
    """
    max_width = max(img.width for img in imgs)
    max_height = max(img.height for img in imgs)

    padded_imgs = []
    for img in imgs:
        # Calculate padding offsets to center the image
        x_padding = (max_width - img.width) // 2
        y_padding = (max_height - img.height) // 2

        padded_img = Image.new("RGB", (max_width, max_height), (255, 255, 255))
        padded_img.paste(img, (x_padding, y_padding))
        padded_imgs.append(padded_img)

    return padded_imgs


def grid_images(imgs: List[Image.Image]) -> Image.Image:
    """
    Given an iterable of images, returns a grid arrangement of them with grid dimensions as square as possible.
    Assumes all images have the same size.
    """
    n = len(imgs)
    grid_w = int(np.ceil(np.sqrt(n)))
    grid_h = int(np.ceil(n / grid_w))

    if len({img.size for img in imgs}) > 1:
        raise ValueError("All images should have the same size.")
    img_w, img_h = imgs[0].size
    grid_img = Image.new("RGB", (grid_w * img_w, grid_h * img_h), (255, 255, 255))

    for i, img in enumerate(imgs):
        iy, ix = i // grid_w, i % grid_w
        pos = (ix * img_w, iy * img_h)
        grid_img.paste(img, pos)

    return grid_img
