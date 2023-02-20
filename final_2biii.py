import cv2
import numpy as np
import os
import scipy.signal as sp
import matplotlib.pyplot as plt
from math import ceil, floor

Z = 0
DIR = "RedChair" if Z else "Office"
images = [
    os.path.join(DIR, f)
    for f in sorted(os.listdir(DIR))
    if f.endswith(".jpg")
]

colour = [cv2.imread(f) for f in images]
gray = [cv2.imread(f, 0) for f in images]


lowThresh = 15
ind = 0
for i in range(len(gray) - 1):
    diff = cv2.absdiff(gray[i], gray[i + 1])

    thresh = cv2.threshold(diff, lowThresh, 255, cv2.THRESH_BINARY)[1]

    col = colour[i].copy()
    col[thresh == 255] = [0, 255, 0]

    cv2.imshow("image", col)
    cv2.imshow("thresh", thresh)

    # cv2.imwrite(f"output/2biii/{DIR}/thresh_{lowThresh}/thresh/image_{ind}.jpg", thresh)
    # cv2.imwrite(f"output/2biii/{DIR}/thresh_{lowThresh}/color/image_{ind}.jpg", col)
    # ind += 1

    if cv2.waitKey(60 if Z else 30) & 0xFF == ord("q"):
        break
