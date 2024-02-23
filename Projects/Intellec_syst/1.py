import cv2
import numpy as np


image = np.zeros((800, 1100, 3), dtype=np.uint8)
image.fill(255)

# CODE

cv2.imshow("Image", image)
cv2.waitKey(0)
