from detec3 import ShapeDetector
import cv2 as cv
import numpy as np
import imutils

#Делаем белый фон
img = np.full((600, 800, 3), 255, dtype=np.uint8) # create

cv.circle(img,(100,100), 63, (0,0,255), -1)#Red круг
cv.circle(img,(250,100), 63, (0,110,255), -1)#Orange круг
cv.circle(img,(400,100), 63, (0,212,255), -1)#Жёлтый круг
cv.circle(img,(550,100), 63, (0,255,89), -1)#Зелёный круг
cv.circle(img,(700,100), 63, (255,145,0), -1)#Blue круг
cv.circle(img,(100,250), 63, (255,0,242), -1)#Purple круг
cv.circle(img,(250,250), 63, (55,0,255), -1)#Red1 круг

#Red равносторонний треугольник
x = -140
y = 250
pts = [(x+170,y+200),(x+280,y+200),(x+225,y+107)]
cv.fillPoly(img, np.array([pts]), (0,0,255))

#Orange треугольник
x+=150
y=260
pts = [(x+165,y+200),(x+290,y+210),(x+225,y+107)]
cv.fillPoly(img, np.array([pts]), (0,110,255))

#жёлтый ромб
x += 300
y += 120
pts = [(x+0,y),(x+70,y+0),(x+105,y+60),(x+35,y+60)]
cv.fillPoly(img, np.array([pts]), (0,212,255))

#зелёный трап
x += 140
y += 60
pts = [(x,y),(x+70,y+0),(x+150,y-70),(x-60,y-70)]
cv.fillPoly(img, np.array([pts]), (0,255,89))

cv.rectangle(img,(x+200,y-100),(x+300,y),(255,145,0),-1)#Blue квадрат

#Purple пятиугольник
x += -400
y += 70
pts = [(x+10,y),(x+20,y-15),(x+80,y+0),(x+105,y+60),(x+35,y+50)]
cv.fillPoly(img, np.array([pts]), (255,0,242))

#Orange парале
x += 150
y += 0
pts = [(x+10,y),(x+80,y+0),(x+105,y+60),(x+35,y+60)]
cv.fillPoly(img, np.array([pts]), (0,110,255))

# Red1 шестиугольник
x += 200
y -=25
pts = [(x+0,y),(x+30,y-40),(x+100,y+0),(x+105,y+60),(x+60,y+90),(x+0,y+60)]
cv.fillPoly(img, np.array([pts]), (55,0,255))

sd = ShapeDetector()

tri = 0 #Счётчики для вывода в терминал
tetr = 0
pent = 0
hex = 0
cir = 0


hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)#Переводим картинку в формат hsv 
masks = {#Словарь цветов
    "Red": ((0, 200, 190), (5, 255, 255)),
    "Orange": ((12, 200, 200), (17, 255, 255)),
    "Yellow": ((20, 200, 200), (32, 255, 255)),
    "Green": ((33, 150, 150), (70, 255, 255)),
    "Blue": ((90, 150, 150), (129, 255, 255)),
    "Purple": ((145, 100, 100), (166, 255, 255)),
    "Red1": ((172, 100, 100), (180, 255, 255)),
}
cnts = 0#Счётчик контуров
trian = ""#Текстовые переменные для терминала
quatro = ""

######Рисуем контуры
#Чб
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#Блюр
blurred = cv.GaussianBlur(gray, (5, 5), 0)
#cv2.imshow("Image",blurred)
#Граница
thresh = cv.threshold(blurred, 250,255,cv.THRESH_BINARY_INV)[1]
# Контуры
conts = cv.findContours(thresh.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
conts=imutils.grab_contours(conts)
for cant in conts:
    cv.drawContours(img, [cant], -1, (0, 190, 255), 2)


#Определяем цвета, форму и площадь
colors = ["Red","Orange","Yellow", "Green","Blue", "Purple","Red1"]
for color in colors:
    mask = cv.inRange(hsv, masks[color][0], masks[color][1])
    # cv.imshow("Image",mask)#дебаг масок
    # cv.waitKey(500)
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2:]
    contours = [cnt for cnt in contours if cv.contourArea(cnt) > 10]
    for c in contours:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
        M = cv.moments(c)
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))
        shape = sd.detect(c)

        area = cv.contourArea(c)
        if ((shape == "equal triangle")or(shape == "isosceles triangle")or (shape == "triangle")):
            tri+=1
            if (trian != ""):
                trian+=", "
            if (shape == "equal triangle"):
                trian += "{} equal triangle - {}".format(color, area)
            elif (shape == "triangle"):
                trian += "{} triangle - {}".format(color, area)
        elif ((shape == "square")or(shape == "trapezoid")or(shape == "diamond")or(shape == "paral")):
            tetr+=1
            if (quatro != ""):
                quatro+=", "
            if (shape == "square"):
                quatro += "{} square - {}".format(color, area)
            elif (shape == "trapezoid"):
                quatro += "{} trapezoid - {}".format(color, area)
            elif (shape == "diamond"):
                quatro += "{} diamond - {}".format(color, area)
            elif (shape == "paral"):
                quatro += "{} paral - {}".format(color, area)
        elif (shape == "pentagon"):
            pent+=1
        elif (shape == "hexagon"):
            hex+=1
        elif (shape == "circle"):
            cir+=1
        # multiply the contour (x, y)-coordinates
        # then draw the contours and the name of the shape on the image
        shape = color + " " + shape
        # cv.drawContours(img, [c], -1, (0, 190, 255), 2)
        cv.circle(img, (cX, cY), 7, (255, 255, 255), -1)
        cv.putText(img, shape, (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3,cv.LINE_AA)
        cv.putText(img, shape, (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1,cv.LINE_AA)
        cv.putText(img, str(area), (cX, cY+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3,cv.LINE_AA)
        cv.putText(img, str(area), (cX, cY+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1,cv.LINE_AA)
    cnts += len(contours)

text = '1142, Solomonov Pavel, number of contours: {}'.format(cnts)
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,text,(10,20), font, 0.7,(5,120,150),4,cv.LINE_AA)
cv.putText(img,text,(10,20), font, 0.7,(30,200,250),2,cv.LINE_AA)
print(text)

print("Triangles: {} ({})".format(tri,trian))
print("Tetragons: {} ({})".format(tetr,quatro))
print("Pentagons: {}".format(pent))
print("Hexagons: {}".format(hex))
print("Circles: {}".format(cir))

# Вывод
cv.imshow("Image",img)
cv.waitKey(0)