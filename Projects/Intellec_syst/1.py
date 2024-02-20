import cv2
import numpy as np


image = np.zeros((512, 512, ), dtype=np.uint8)
image.fill(255)

# CODE

cv2.imshow("Image", image)
cv2.waitKey(0)
