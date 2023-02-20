import cv2
import numpy as np
import os
import scipy.signal as sp
import matplotlib.pyplot as plt

DIR = "RedChair"
images = [
    os.path.join(DIR, f)
    for f in sorted(os.listdir(DIR))
    if f.endswith(".jpg")
]

colour = [cv2.imread(f) for f in images]
gray = [cv2.imread(f, 0) for f in images]

gray = [cv2.GaussianBlur(i, (5, 5), 5) for i in gray]

f = []
fg = []

p1DH = np.array([[-0.5, 0, 0.5]])
p1DV = p1DH.T
p3DH = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
p3DV = p3DH.T
p5DH = np.array([[-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]])
p5DV = p5DH.T

for i in range(len(gray) - 1):
    diff = cv2.absdiff(gray[i], gray[i + 1])

    # kernel = np.ones((3, 3), np.float32) / 9              # Medium success with this one
    kernel = np.ones((5, 5), np.float32) / 25             # Low success with this one

    diff1 = sp.convolve2d(diff, p1DH, mode="same", boundary="fill", fillvalue=0)
    diff2 = sp.convolve2d(diff, p1DV, mode="same", boundary="fill", fillvalue=0)
    diff = diff1 + diff2

    diff = sp.convolve2d(diff, kernel, mode="same", boundary="fill", fillvalue=0)

    diff = cv2.GaussianBlur(diff, (5,5), 5)

    diff = np.uint8(diff)
    # thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)[1]

    fg.append(thresh)

    col = colour[i].copy()
    col[thresh == 255] = [0, 255, 0]

    f.append(col)
    cv2.imshow("image", col)
    cv2.imshow("thresh", thresh)
    if cv2.waitKey(60) & 0xFF == ord("q"):
        break

# height, width, layers = f[0].shape
# video = cv2.VideoWriter(
#     f"minorSuccess_colour.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (width, height)
# )

# gray_video = cv2.VideoWriter(
#     f"minorSuccess_gray.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (width, height), 0
# )

# for image in f:
#     video.write(np.uint8(image))

# for image in fg:
#     gray_video.write(np.uint8(image))
