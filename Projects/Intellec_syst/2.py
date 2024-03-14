import cv2
import numpy as np
import imutils

# пустое изображение, заполненное белым
image = np.zeros((700, 1100, 3), dtype=np.uint8)
image.fill(255)


# данные для вывода фигур
layers_delta = 300

width = 150
spacing = 100

first_layer = 150
second_layer = first_layer + layers_delta

first_fig = width + width
second_fig = first_fig + spacing + width
third_fig = second_fig + spacing + width
forth_fig = third_fig + spacing

spacingS = int(spacing + width / 2)
second_figS = spacingS + spacing + width + 100
third_figSC = int(second_figS + width / 2) + 100


# вывод фигур
points = np.array([[spacing + int(width / 2), first_layer - 50],
                   [spacing - 10, first_layer + int(width / 2)],
                   [spacing + int(width / 2), first_layer + width + 50],
                   [spacing + width + 10, first_layer + int(width / 2)]])
cv2.drawContours(image, [points], 0, (100, 0, 100), -1)
cv2.rectangle(image, (first_fig + spacing, first_layer), (second_fig, first_layer + width), (0, 255, 0), -1)
points = np.array([[600, 350], [800, 350], [700, 150]])
cv2.drawContours(image, [points], 0, (255, 0, 0), -1)
points = np.array([[forth_fig + 30, first_layer],
                   [forth_fig - 30, first_layer + width],
                   [forth_fig + width + 30, first_layer + width],
                   [forth_fig + width - 10, first_layer]])
cv2.drawContours(image, [points], 0, (255, 255, 0), -1)


cv2.circle(image, (spacingS, second_layer + int(width / 2)), int(width / 2), (255, 0, 100), -1)
points = np.array([[spacingS + spacingS + int(width / 2), second_layer],
                   [spacingS + spacingS - 15, second_layer + width - 100],
                   [spacingS + spacingS, second_layer + width],
                   [spacingS + spacingS + width, second_layer + width],
                   [spacingS + spacingS + width + 15, second_layer + width - 100]])
cv2.drawContours(image, [points], 0, (255, 0, 0), -1)
cv2.circle(image, (second_figS + spacingS, second_layer + int(width / 2)), int(width / 2), (150, 50, 255), -1)
points = np.array([[third_figSC + spacingS + int(width / 2), second_layer],
                   [third_figSC + spacingS - 15, second_layer + width - 100],
                   [third_figSC + spacingS, second_layer + width],
                   [third_figSC + spacingS + width, second_layer + width],
                   [third_figSC + spacingS + width + 15, second_layer + width - 100]])
cv2.drawContours(image, [points], 0, (50, 0, 150), -1)
points = np.array([[700, 10], [1000, 50], [790, 100]])
cv2.drawContours(image, [points], 0, (50, 60, 100), -1)


# Нахождение контуров
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

result = image.copy()


# Счетчики для каждого типа фигур
triangle_count = 0
square_count = 0
rectangle_count = 0
pentagon_count = 0
hexagon_count = 0
circle_count = 0
unknown_count = 0
rhombus_count = 0
trapezoid_count = 0

for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        Shape = "Unknown"
        Perimeter = cv2.arcLength(contour, True)
        Apprx = cv2.approxPolyDP(contour, 0.04 * Perimeter, True)

        match len(Apprx):
            case 3:
                Shape = "Triangle"
                triangle_count += 1
            case 4:
                (x, y, w, h) = cv2.boundingRect(Apprx)
                if w == h:
                    Shape = "Square"
                    square_count += 1
                else:
                    Shape = "Rectangle"
                    rectangle_count += 1
            case 5:
                Shape = "Pentagon"
                pentagon_count += 1
            case _:
                Shape = "Circle"
                circle_count += 1

        cv2.putText(result, Shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.circle(result, (cX, cY), 2, (0, 0, 0), -1)
        cv2.drawContours(result, [contour], -1, (0, 0, 0), 2)


# Добавляем текст с информацией о количестве объектов
text = f"I am Konovalov Anton from group 1142 and I found {len(contours)} Objects"
cv2.putText(result, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


# Выводим количество каждого типа фигур
print(text)
print("Total number of figures:", len(contours))
print("Triangles –", triangle_count)
print("Squares –", square_count)
print("Trapezoids –", trapezoid_count)
print("Rhombuses –", rhombus_count)
print("Rectangles –", rectangle_count)
print("Pentagons –", pentagon_count)
print("Hexagons –", hexagon_count)
print("Circles –", circle_count)


# Отображаем результат
cv2.imshow("Image", result)
cv2.waitKey(0)
