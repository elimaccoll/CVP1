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


ind = 0
for i in range(len(gray) - 1):
    diff = cv2.absdiff(gray[i], gray[i + 1])

    # Thresh 20
    thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)[1]

    col = colour[i].copy()
    col[thresh == 255] = [0, 255, 0]

    cv2.imshow("image", col)
    cv2.imshow("thresh", thresh)

    # cv2.imwrite(f"output/{DIR}/diff/image_{ind}.jpg", diff)
    # cv2.imwrite(f"output/{DIR}/diffThresh/image_{ind}.jpg", thresh)
    # cv2.imwrite(f"output/{DIR}/diffThreshColor/image_{ind}.jpg", col)
    # ind += 1

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break
