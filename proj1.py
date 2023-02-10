# Detecting motion in a series of images.
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import scipy.signal as sp


# Get images from the folder
def getImagesFromDir(dir: str, ext: str = ".jpg"):
    original = []
    gray = []
    for file in sorted(os.listdir(dir)):
        if not file.endswith(ext):
            continue
        print(f"Processing {file}", end="\r")
        image = cv2.imread(f"{dir}/{file}")
        grayImage = cv2.imread(f"{dir}/{file}", 0)
        original.append(image)
        gray.append(grayImage)
    return original, gray


# Convolve the image with a 1D kernel
def convolve1D(image, kernel):
    return sp.convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0)


DIR = "RedChair"
# DIR = "Office"
images, grays = getImagesFromDir(DIR)

filter1D = np.array([[-0.5, 0, 0.5]])
filter2D_3 = np.ones((3, 3)) / 9

final = []
for ig in range(len(grays) - 1):
    # Get the difference between the current image and the next image
    diff = np.abs(grays[ig + 1] - grays[ig])

    # Convolve the image with a 1D kernel
    diff = convolve1D(diff, filter1D)

    # Create a 1D derivative of a Gaussian kernel with sigma = 1
    # kernel = np.random.normal(0, 1, (3,3))

    # Convolve the image with a 2D kernel
    # diff = convolve1D(diff, kernel)

    # Threshold the image
    _, diff = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    og = images[ig].copy()
    og[diff == 255] = [0, 0, 255]

    final.append(og)

# Save final images as video file
height, width, layers = final[0].shape
video = cv2.VideoWriter(
    f"2D_3_woGauss.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 5, (width, height)
)

for image in final:
    video.write(np.uint8(image))

video.release()

# for image in final:
#     cv2.imshow("Image", image)
#     if cv2.waitKey(200) & 0xFF == ord("q"):
#         break
