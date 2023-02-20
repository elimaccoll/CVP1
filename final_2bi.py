import cv2
import numpy as np
import os
import scipy.signal as sp
import matplotlib.pyplot as plt
from math import ceil, floor

def createGaussian(sigma):
    size = ceil(5 * sigma)
    s2 = sigma * sigma
    mask = np.zeros((size, size))

    low = floor(size/2)
    high = ceil(size/2)
    for x in range(-low, high):
        for y in range(-low, high):
            mask[x+low, y+low] = np.exp((-x**2 - y**2) / (2 * s2))
    mask /= np.min(mask)
    mask = np.round(mask)
    return mask


Z = 0
DIR = "RedChair" if Z else "Office"
images = [
    os.path.join(DIR, f)
    for f in sorted(os.listdir(DIR))
    if f.endswith(".jpg")
]

colour = [cv2.imread(f) for f in images]
gray = [cv2.imread(f, 0) for f in images]

p1DH = np.array([[-0.5, 0, 0.5]])

tSigma = 0.6
g1DH = np.array([createGaussian(tSigma)[0]])

ind = 0
for i in range(len(gray)):
    diff = sp.convolve2d(gray[i], p1DH, mode="same", boundary="fill", fillvalue=0)
    diff = sp.convolve2d(diff, g1DH, mode="same", boundary="fill", fillvalue=0)

    # Thresh 20
    thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)[1]

    col = colour[i].copy()
    col[thresh == 255] = [0, 255, 0]

    cv2.imshow("image", col)
    cv2.imshow("thresh", thresh)

    # cv2.imwrite(f"output/2bi/{DIR}/sigma0_1/thresh/image_{ind}.jpg", thresh)
    # cv2.imwrite(f"output/2bi/{DIR}/sigma0_1/color/image_{ind}.jpg", col)
    # ind += 1

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break
