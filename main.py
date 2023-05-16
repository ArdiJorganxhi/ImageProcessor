import functions
import cv2

loop = True

img = cv2.imread('YOUR_IMAGE_FILE')

while loop:
    print("Welcome to Image Processor: \n"
          "1. Flip Image Vertically \n"
          "2. Rotate 90 degrees counterclockwise \n"
          "3. Rotate 90 degrees clockwise \n"
          "4. Resize Image \n"
          "5. Negative Transformation \n"
          "6. Histogram Equalization (Grayscale) \n"
          "7. NxN Average Filter \n"
          "8. Red Highlight \n"
          "9. Exit")

    user = int(input())

    if user == 1:
        img = functions.flip_image_vertically_function(img)
    elif user == 2:
        img = functions.rotate_image_90degrees_counterclockwise_function(img)
    elif user == 3:
        img = functions.rotate_image_90degrees_clockwise_function(img)
    elif user == 4:
        img = functions.resize_image_function(img)
    elif user == 5:
        img = functions.negative_transformation_function(img)
    elif user == 6:
        img = functions.histogram_equalization_function(img)
    elif user == 7:
        user_size = int(input("Please enter a size for NxN Average Filter:"))
        img = functions.nxn_average_filter_function(img, user_size)
    elif user == 8:
        img = functions.red_highlighted_function(img)
    elif user == 9:
        cv2.imwrite("output_image.png", img)
        loop = False
