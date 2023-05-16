from algorithms import histogram_equalization, nxn_average_filter, red_highlight_algorithm
from algorithms import flip_images_vertically, rotate_image_90degrees_counterclockwise, rotate_image_90degrees_clockwise
from algorithms import resize_image, negative_transformation
import numpy as np

def flip_image_vertically_function(img):
    vertically_image = flip_images_vertically(img)
    return vertically_image

def rotate_image_90degrees_counterclockwise_function(img):
    counterclockwise_image = rotate_image_90degrees_counterclockwise(img)
    return counterclockwise_image

def rotate_image_90degrees_clockwise_function(img):
    clockwise_image = rotate_image_90degrees_clockwise(img)
    return clockwise_image

def resize_image_function(img):
    resized_image = resize_image(img)
    return resized_image

def negative_transformation_function(img):
    negative_image = negative_transformation(img)
    return negative_image

def histogram_equalization_function(img):
    equalized_image_with_algorithm = histogram_equalization.algorithm(img)
    equalized_image_with_algorithm = np.uint8(equalized_image_with_algorithm)
    return equalized_image_with_algorithm



def nxn_average_filter_function(img, kernel_size):
    nxn_averaged_image_with_algorithm = nxn_average_filter.algorithm(img, size=kernel_size)
    nxn_averaged_image_with_algorithm = np.uint8(nxn_averaged_image_with_algorithm)
    return nxn_averaged_image_with_algorithm


def red_highlighted_function(img):
    red_highlighted_with_algorithm = red_highlight_algorithm(img)
    return red_highlighted_with_algorithm
