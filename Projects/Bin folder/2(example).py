import cv2
import imutils
import numpy as np

image = np.zeros((850, 1000, 3), np.uint8)
image.fill(255)

# Рисуем новые фигуры
cv2.circle(image, (150, 180), 70, (255, 0, 0), -1)  # Круг
points_triangle = np.array([[50, 700], [250, 700], [150, 500]])
cv2.drawContours(image, [points_triangle], 0, (180, 200, 100), -1)  # Равносторонний треугольник
points_arbitrary_triangle = np.array([[400, 700], [800, 700], [500, 570]])
cv2.drawContours(image, [points_arbitrary_triangle], 0, (255, 0, 255), -1)  # Произвольный треугольник
points_rhombus = np.array([[800, 80], [850, 0], [900, 80], [850, 160]])
cv2.drawContours(image, [points_rhombus], 0, (255, 255, 0), -1)  # Ромб
points_trapezoid = np.array([[50, 300], [250, 300], [350, 450], [50, 450]])
cv2.drawContours(image, [points_trapezoid], 0, (0, 0, 255), -1)  # Трапеция
cv2.rectangle(image, (400, 300), (600, 500), (0, 255, 255), -1)  # Квадрат
points_pentagon = np.array([[750, 250], [900, 250], [975, 350], [875, 450], [775, 450]])
cv2.drawContours(image, [points_pentagon], 0, (128, 0, 128), -1)  # Пятиугольник
cv2.circle(image, (600, 180), 60, (120, 150, 40), -1)  # Круг
cv2.rectangle(image, (950, 600), (700, 500), (65, 16, 105), -1)  # бордовый прямоугольник
points_octagon = np.array([[250, 750], [300, 800], [350, 800], [400, 750], [350, 700], [300, 700]])
cv2.drawContours(image, [points_octagon], 0, (0, 128, 128), -1)  # Восьмиугольник

# Находим контуры новых объектов
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

# Рисуем контуры и находим центры на копии фона
output = image.copy()

# Счетчики для каждого типа фигур
triangle_count = 0
square_count = 0
rectangle_count = 0
pentagon_count = 0
hexagon_count = 0
circle_count = 0
unknown_count = 0

for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        shape = "Unknown"
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        if len(approx) == 3:
            shape = "Triangle"
            triangle_count += 1
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            if ar >= 0.95 and ar <= 1.05:
                shape = "Square"
                square_count += 1
            else:
                shape = "Rectangle"
                rectangle_count += 1
        elif len(approx) == 5:
            shape = "Pentagon"
            pentagon_count += 1
        elif len(approx) == 6:
            shape = "Hexagon"
            hexagon_count += 1
        else:
            shape = "Circle"
            circle_count += 1

        cv2.putText(output, shape, (cX - 50, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.circle(output, (cX, cY), 3, (0, 0, 0), -1)
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)  # Зеленый цвет

# Выводим количество каждого типа фигур
print("Общее количество фигур:", len(contours))
print("Треугольников -", triangle_count)
print("Квадратов -", square_count)
print("четырехугольников -", rectangle_count)
print("Пятиугольников -", pentagon_count)
print("Шестиугольников -", hexagon_count)
print("Кругов -", circle_count)

# Добавляем текст с информацией о количестве объектов
text = "I am Egor Mukhin, a student of group 1141 and I found {} Objects".format(len(contours))
cv2.putText(output, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

# Отображаем результат
cv2.imshow("Result", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
