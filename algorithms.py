import numpy as np
import cv2


def flip_images_vertically(img):
    height, width = img.shape[:2]
    flipped_img = np.zeros_like(img)
    for i in range(height):
        flipped_img[i, :] = img[height - i - 1, :]

    return flipped_img


def rotate_image_90degrees_counterclockwise(img):
    height, width = img.shape[:2]
    rotated_img = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            rotated_img[j][height - i - 1] = img[i][j]

    return rotated_img


def rotate_image_90degrees_clockwise(img):
    height, width = img.shape[:2]
    rotated_img = [[0 for _ in range(width)] for _ in range(height)]

    for i in range(height):
        for j in range(width):
            rotated_img[width - j - 1][i] = img[i][j]

    rotated_img = cv2.UMat(np.asarray(rotated_img))

    return rotated_img


def resize_image(img):
    height, width = img.shape[:2]

    new_height = height // 2
    new_width = width // 2

    img_half = [[0 for _ in range(new_width)] for _ in range(new_height)]
    sx = width / new_width
    sy = height / new_height

    for i in range(new_height):
        for j in range(new_width):
            x = int(j * sx)
            y = int(i * sy)
            img_half[i][j] = img[y][x]

    img_half = np.array(img_half)

    return img_half


def negative_transformation(img):
    height, width = img.shape[:2]
    neg_img = img.copy()

    for i in range(height):
        for j in range(width):
            r, g, b = img[i, j]
            neg_r, neg_g, neg_b = 255 - r, 255 - g, 255 - b
            neg_img[i, j] = (neg_r, neg_g, neg_b)

    return neg_img


def histogram_equalization(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = [0] * 256
    for row in img:
        for pixel in row:
            hist[pixel] += 1

    # Compute CDF for Histogram
    cdf = [hist[0]]
    for i in range(1, 256):
        cdf.append(cdf[i - 1] + hist[i])

    # Normalize CDF
    cdf_normalized = [round((val - cdf[0]) * 255 / (len(img) * len(img[0]) - cdf[0])) for val in cdf]

    # Histogram Equalization of Images
    equalized_image = []
    for row in img:
        equalized_row = []
        for pixel in row:
            equalized_pixel = cdf_normalized[pixel]
            equalized_row.append(equalized_pixel)
        equalized_image.append(equalized_row)

    return equalized_image


def nxn_average_filter(img, size):
    img = np.asarray(img)
    # Define the filtered image as an empty list
    filtered_image = []

    # Iterate over the rows of the original image
    for i in range(size // 2, len(img) - size // 2):
        row = []
        # Iterate over the columns of the original image
        for j in range(size // 2, len(img[i]) - size // 2):
            sum = 0
            # Iterate over the pixels in the n x n neighborhood
            for k in range(i - size // 2, i + size // 2 + 1):
                for l in range(j - size // 2, j + size // 2 + 1):
                    sum += img[k][l]
            row.append(sum // (size * size))
        filtered_image.append(row)

    # Return the filtered image as a NumPy array
    return np.array(filtered_image)


def red_highlight_algorithm(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define a range of red color in HSV
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    red_mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Combine the red masks
    red_mask = red_mask1 + red_mask2

    # Apply the red mask to the original image
    red_regions = cv2.bitwise_and(img, img, mask=red_mask)

    # Create a mask for the non-red regions
    non_red_mask = cv2.bitwise_not(red_mask)

    # Apply the non-red mask to the original image and convert to grayscale
    gray = cv2.cvtColor(cv2.bitwise_and(img, img, mask=non_red_mask), cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    result = cv2.add(red_regions, gray)

    return result


