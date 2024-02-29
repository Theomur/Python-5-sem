import cv2
import numpy as np
import imutils
from random import randint

# пустое изображение, заполненное белым
image = np.zeros((700, 1100, 3), dtype=np.uint8)
image.fill(255)


# данные для вывода фигур
layers_delta = 300

width = 150
spacing = 100

first_layer = 150
second_layer = int(first_layer + layers_delta + width / 2)

first_fig = width + width
second_fig = first_fig + spacing + width
third_fig = second_fig + spacing + width

spacingC = int(spacing + width / 2)
second_figC = int(spacingC + width / 2)
third_figC = int(second_figC + width / 2)


def random():
    return randint(-70, 70)


# вывод фигур
cv2.rectangle(image, (spacing, first_layer + random()), (first_fig, first_layer + width + random()), (100, 0, 100), -1)
cv2.rectangle(image, (first_fig + spacing, first_layer + random()), (second_fig, first_layer + width + random()), (0, 255, 0), -1)
cv2.rectangle(image, (second_fig + spacing, first_layer + random()), (third_fig, first_layer + width + random()), (255, 0, 0), -1)

cv2.circle(image, (spacingC, second_layer + random()), int(width / 2), (0, 0, 0), -1)
cv2.circle(image, (second_figC + spacingC + random(), second_layer + random()), int(width / 2), (0, 255, 255), -1)
cv2.circle(image, (third_figC + spacingC * 3 + random(), second_layer + random()), int(width / 2), (255, 255, 0), -1)


# нахождение контуров
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)


# вывод контуров
result = image.copy()
cv2.drawContours(result, contours, -1, (160, 32, 240), 3)

# вывод текста
text = f"I am Konovalov Anton, a student of group 1142 and I found {len(contours)} Objects :)"
cv2.putText(result, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
print(text)

# круг и квадрат
cv2.rectangle(result, (int(1100 / 2 - 5), int(700 / 2 - 5)), (int(1100 / 2 + 5), int(700 / 2 + 5)), (0, 165, 255), -1)
cv2.circle(result, (0, 695), 10, (0, 165, 255), -1)

# вывод результата
cv2.imshow("Image", result)
cv2.waitKey(0)
