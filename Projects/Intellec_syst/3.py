import cv2
import numpy as np
import imutils
import math

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

colour_ranges = {
    "Red": ((0, 200, 190), (5, 255, 255)),
    "Orange": ((12, 200, 200), (17, 255, 255)),
    "Yellow": ((20, 200, 200), (32, 255, 255)),
    "Green": ((33, 150, 150), (70, 255, 255)),
    "Blue": ((90, 150, 150), (129, 255, 255)),
    "Pink": ((145, 100, 100), (166, 255, 255))
}

colours = {
    "Red": (0, 0, 255),
    "Orange": (0, 100, 255),
    "Yellow": (0, 255, 255),
    "Green": (0, 200, 0),
    "Blue": (255, 0, 0),
    "Pink": (255, 0, 242)
}


# вывод фигур
points = np.array([[spacing + int(width / 2), first_layer - 50],
                   [spacing - 10, first_layer + int(width / 2)],
                   [spacing + int(width / 2), first_layer + width + 50],
                   [spacing + width + 10, first_layer + int(width / 2)]])
cv2.drawContours(image, [points], 0, colours["Red"], -1)

cv2.rectangle(image, (first_fig + spacing, first_layer), (second_fig, first_layer + width), colours["Orange"], -1)

points = np.array([[600, 350], [800, 350], [700, 150]])
cv2.drawContours(image, [points], 0, colours["Yellow"], -1)

points = np.array([[forth_fig + 30, first_layer],
                   [forth_fig - 30, first_layer + width],
                   [forth_fig + width + 30, first_layer + width],
                   [forth_fig + width - 10, first_layer]])
cv2.drawContours(image, [points], 0, colours["Green"], -1)


cv2.circle(image, (spacingS, second_layer + int(width / 2)), int(width / 2), colours["Blue"], -1)

points = np.array([[spacingS + spacingS + int(width / 2), second_layer],
                   [spacingS + spacingS - 15, second_layer + width - 100],
                   [spacingS + spacingS, second_layer + width],
                   [spacingS + spacingS + width, second_layer + width],
                   [spacingS + spacingS + width + 15, second_layer + width - 100]])
cv2.drawContours(image, [points], 0, colours["Orange"], -1)

cv2.circle(image, (second_figS + spacingS, second_layer + int(width / 2)), int(width / 2), colours["Pink"], -1)

points = np.array([[third_figSC + spacingS + int(width / 2), second_layer],
                   [third_figSC + spacingS - 15, second_layer + width - 100],
                   [third_figSC + spacingS, second_layer + width],
                   [third_figSC + spacingS + width, second_layer + width],
                   [third_figSC + spacingS + width + 15, second_layer + width - 100]])
cv2.drawContours(image, [points], 0, colours["Yellow"], -1)

points = np.array([[700, 10], [1000, 50], [790, 100]])
cv2.drawContours(image, [points], 0, colours["Red"], -1)

points = np.array([[300, 400], [410, 400], [355, 307]])
cv2.drawContours(image, [points], 0, colours["Green"], -1)


# Нахождение контуров
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
conts = imutils.grab_contours(contours)

result = image.copy()


# Счетчики для каждого типа фигур
triangle_count = 0
equal_triangle_count = 0
isosceles_triangle_count = 0
square_count = 0
rectangle_count = 0
pentagon_count = 0
circle_count = 0
unknown_count = 0
rhombus_count = 0
trapezoid_count = 0

all_areas = {
    "equal triangle": [],
    "isosceles triangle": [],
    "square": [],
    "rhombus": [],
    "rectangle": [],
    "trapezoid": [],
    "Pentagon": [],
    "Circle": []
}


hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)


# вспомогательные функции
def get_angle(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p1[0] - p3[0], p1[1] - p3[1]])
    v1_unit = unit_vector(v1)
    v2_unit = unit_vector(v2)
    radians = np.arccos(np.clip(np.dot(v1_unit, v2_unit), -1, 1))
    return math.degrees(radians)


def unit_vector(v):
    return v / np.linalg.norm(v)


def loose_equ(a, b, wiggle):
    return abs(a - b) <= wiggle


# логика
colors_cycle = ["Red", "Orange", "Yellow", "Green", "Blue", "Pink"]
amount = len(conts)
for colour in colors_cycle:
    mask = cv2.inRange(hsv, colour_ranges[colour][0], colour_ranges[colour][1])
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            Shape = "Unknown"
            Perimeter = cv2.arcLength(contour, True)
            Apprx = cv2.approxPolyDP(contour, 0.04 * Perimeter, True)

            Square_i = cv2.contourArea(contour)

            match len(Apprx):
                case 3:
                    angle1 = get_angle(Apprx[0][0], Apprx[1][0], Apprx[2][0])
                    angle2 = get_angle(Apprx[1][0], Apprx[2][0], Apprx[0][0])
                    angle3 = get_angle(Apprx[2][0], Apprx[0][0], Apprx[1][0])

                    if (loose_equ(angle1, 60, 2) and loose_equ(angle2, 60, 2)):
                        Shape = "equal triangle"
                        equal_triangle_count += 1
                        triangle_count += 1
                    elif (loose_equ(angle1, angle2, 3) or loose_equ(angle1, angle3, 3) or loose_equ(angle3, angle2, 3)):
                        Shape = "isosceles triangle"
                        isosceles_triangle_count += 1
                        triangle_count += 1
                    else:
                        Shape = "triangle"
                        triangle_count += 1
                case 4:
                    (x, y, w, h) = cv2.boundingRect(Apprx)
                    ar = w / float(h)

                    angle1 = math.degrees(math.atan2(Apprx[1][0][0] - Apprx[0][0][0], Apprx[1][0][1] - Apprx[0][0][1]))
                    angle2 = math.degrees(math.atan2(Apprx[1][0][0] - Apprx[2][0][0], Apprx[1][0][1] - Apprx[2][0][1]))
                    angle3 = math.degrees(math.atan2(Apprx[2][0][0] - Apprx[3][0][0], Apprx[2][0][1] - Apprx[3][0][1]))
                    angle4 = math.degrees(math.atan2(Apprx[0][0][0] - Apprx[3][0][0], Apprx[0][0][1] - Apprx[3][0][1]))

                    if 0.95 <= ar <= 1.05:
                        Shape = "square"
                        square_count += 1
                    else:
                        if angle3 - 1 <= angle1 <= angle3 + 1 and angle4 - 1 <= angle2 <= angle4 + 1:
                            o1 = math.sqrt((Apprx[1][0][0] - Apprx[0][0][0]) ** 2 + (Apprx[1][0][1] - Apprx[0][0][1]) ** 2)
                            o2 = math.sqrt((Apprx[2][0][0] - Apprx[1][0][0]) ** 2 + (Apprx[2][0][1] - Apprx[1][0][1]) ** 2)
                            o3 = math.sqrt((Apprx[3][0][0] - Apprx[2][0][0]) ** 2 + (Apprx[3][0][1] - Apprx[2][0][1]) ** 2)
                            o4 = math.sqrt((Apprx[0][0][0] - Apprx[3][0][0]) ** 2 + (Apprx[0][0][1] - Apprx[3][0][1]) ** 2)
                            if (o2 - 2 <= o1 <= o2 + 2) and (o3 - 2 <= o4 <= o3 + 2) and (o4 - 2 <= o1 <= o4 + 2):
                                Shape = "rhombus"
                                rhombus_count += 1
                            else:
                                Shape = "rectangle"
                                rectangle_count += 1
                        elif angle3 - 1 <= angle1 <= angle3 + 1 or angle4 - 1 <= angle2 <= angle4 + 1:
                            Shape = "trapezoid"
                            trapezoid_count += 1
                case 5:
                    Shape = "Pentagon"
                    pentagon_count += 1
                case _:
                    Shape = "Circle"
                    circle_count += 1

            Colour_info = f"{colour}, {Square_i}"
            if (Shape != 'triangle'):
                all_areas[Shape].append(Square_i)

            cv2.drawContours(result, [contour], -1, (0, 0, 0), 2)
            cv2.putText(result, Shape, (cX + 5, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            cv2.putText(result, Colour_info, (cX + 5, cY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            cv2.circle(result, (cX, cY), 2, (0, 0, 0), -1)


# Добавляем текст с информацией о количестве объектов
text = f"I am Konovalov Anton from group 1142 and I found {amount} Objects"
cv2.putText(result, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


# Выводим количество каждого типа фигур
print(text)
print("Total number of figures:", amount)
print("Triangles –",
      triangle_count,
      f"(equal triangles = {equal_triangle_count}, areas: {all_areas['equal triangle']}",
      f"; isosceles triangles = {isosceles_triangle_count}, areas: {all_areas['isosceles triangle']}")
print("Squares –", square_count, "; areas:", all_areas["square"])
print("Trapezoids –", trapezoid_count, "; areas:", all_areas["trapezoid"])
print("Rhombuses –", rhombus_count, "; areas:", all_areas["rhombus"])
print("Rectangles –", rectangle_count, "; areas:", all_areas["rectangle"])
print("Pentagons –", pentagon_count, "; areas:", all_areas["Pentagon"])
print("Circles –", circle_count, "; areas:", all_areas["Circle"])


# Отображаем результат
cv2.imshow("Image", result)
cv2.waitKey(0)
