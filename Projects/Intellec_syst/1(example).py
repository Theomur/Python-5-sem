import cv2
import imutils
import numpy as np

image = np.zeros((850, 1000, 3), np.uint8)
image.fill(255)

cv2.rectangle(image, (200, 200), (400, 400), (0, 0, 255), -1)  # Красный квадрат
cv2.rectangle(image, (600, 350), (800, 400), (212, 255, 127), -1)  # берюзовый прямоугольник
cv2.ellipse(image, (330, 100), (250, 20), 0, 0, 360, (255, 192, 203), -1)  # Розовый эллипс
cv2.rectangle(image, (160, 600), (500, 500), (65, 16, 105), -1)  # бордовый прямоугольник
cv2.circle(image, (800, 600), 100, (0, 255, 255), -1)  # Желтый круг
cv2.circle(image, (100, 200), 40, (30, 105, 210), -1)  # оранжевый круг
cv2.circle(image, (500, 200), 68, (30, 105, 210), -1)  # оранжевый круг
cv2.rectangle(image, (475, 400), (525, 450), (50, 205, 154), -1)  # Красный квадрат по центру

# Находим контуры новых объектов
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

# Рисуем контуры на копии фона
output = image.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 3)

# Добавляем текст с информацией о количестве объектов
text = "I am Egor Mukhin, a student of group 1141 and I found {} Objects".format(len(contours))
cv2.putText(output, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
print(text)

# Отображаем результат
cv2.imshow("Result", output)
cv2.waitKey(0)
