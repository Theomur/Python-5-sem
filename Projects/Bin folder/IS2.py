from detec import ShapeDetector
import cv2
import imutils
import numpy as np


#Делаем белый фон
img = np.full((600, 800, 3), 255, dtype=np.uint8) # create
# Фигуры
cv2.rectangle(img,(584,150),(510,278),(0,255,0),-1)#прямоугольник
cv2.rectangle(img,(600,150),(700,250),(30,190,250),-1)#квадрат
cv2.circle(img,(100,300), 63, (0,0,200), -1)#круг

#равносторонний треугольник
x = 200
y = 200
pts = [(x+170,y+200),(x+280,y+200),(x+225,y+107)]
cv2.fillPoly(img, np.array([pts]), (70,150,0))

#равнобедренный треугольник
x-=140
y-=15
pts = [(x+165,y+200),(x+290,y+200),(x+225,y+107)]
cv2.fillPoly(img, np.array([pts]), (150,150,0))

#треугольник
x+=40
y+=110
pts = [(x+165,y+200),(x+290,y+210),(x+225,y+107)]
cv2.fillPoly(img, np.array([pts]), (150,80,0))

#трап
x = 140
y = 200
pts = [(x,y),(x+70,y+0),(x+150,y-70),(x-60,y-70)]
cv2.fillPoly(img, np.array([pts]), (50,150,90))

#ромб
x = 340
y = 100
pts = [(x+0,y),(x+70,y+0),(x+105,y+60),(x+35,y+60)]
cv2.fillPoly(img, np.array([pts]), (50,150,36))

#ромб
x = 340
y = 200
pts = [(x+0,y),(x+30,y-30),(x+105,y+60),(x+35,y+60)]
cv2.fillPoly(img, np.array([pts]), (50,150,36))

#парале
x = 440
y = 80
pts = [(x+10,y),(x+80,y+0),(x+105,y+60),(x+35,y+60)]
cv2.fillPoly(img, np.array([pts]), (50,200,0))

#пятиугольник
x = 540
y = 80
pts = [(x+10,y),(x+20,y-15),(x+80,y+0),(x+105,y+60),(x+35,y+50)]
cv2.fillPoly(img, np.array([pts]), (50,150,140))

#шестиугольник
x = 500
y = 410
pts = [(x+0,y),(x+30,y-40),(x+100,y+0),(x+105,y+60),(x+60,y+90),(x+0,y+60)]
cv2.fillPoly(img, np.array([pts]), (50,150,250))

#семиугольник
x = 600
y = 310
pts = [(x+0,y),(x+30,y-40),(x+70,y-40),(x+100,y+0),(x+105,y+60),(x+60,y+90),(x+0,y+60)]
cv2.fillPoly(img, np.array([pts]), (250,150,200))


# cv2.imshow("Image", img)
# cv2.waitKey(0)

#Чб
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#Блюр
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#cv2.imshow("Image",blurred)
#Граница
thresh = cv2.threshold(blurred, 250,255,cv2.THRESH_BINARY_INV)[1]
#Контуры
cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)

sd = ShapeDetector()

equtri = 0
isotri = 0
tri = 0

sqr = 0
rect = 0
trap = 0
dia = 0
paral = 0
tetr = 0
fiv = 0
six = 0
sev = 0
cir = 0


for c in cnts:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]))
    cY = int((M["m01"] / M["m00"]))
    shape = sd.detect(c)

    if (shape == "equal triangle"):
        equtri+=1
    elif (shape == "isosceles triangle"):
        tri+=1
    elif (shape == "triangle"):
        isotri+=1
    elif (shape == "square"):
        sqr+=1
    elif (shape == "rectangle"):
        rect+=1
    elif (shape == "trapezoid"):
        trap+=1
    elif (shape == "diamond"):
        dia+=1
    elif (shape == "paral"):
        paral+=1
    elif (shape == "tetragon"):
        tetr+=1
    elif (shape == "pentagon"):
        fiv+=1
    elif (shape == "hexagon"):
        six+=1
    elif (shape == "heptagon"):
        sev+=1
    elif (shape == "circle"):
        cir+=1
    # multiply the contour (x, y)-coordinates
    # then draw the contours and the name of the shape on the image
    cv2.drawContours(img, [c], -1, (0, 190, 255), 2)
    cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(img, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
    # show the output image


cv2.imshow("Image", img)
cv2.waitKey(0)

# # Финалка
text = '1142, Solomonov Pavel, number of contours: {}'.format(len(cnts))
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,text,(10,20), font, 0.7,(30,200,250),2,cv2.LINE_AA)
print(text)
print("Triangles: {} (regular - {}, isosceles - {}, equal - {})".format(tri+isotri+equtri,tri,isotri,equtri))
print("Tetragons: {} (regular - {}, square - {}, rectangle - {}, trapezoid - {}, diamond - {}, parallelogram - {})".format(tetr+sqr+rect+trap+dia+paral,tetr,sqr,rect,trap,dia,paral))
print("Pentagons: {}".format(fiv))
print("Hexagons: {}".format(six))
print("Heptagons: {}".format(sev))
print("Circles: {}".format(cir))

cv2.imshow("Image", img)
cv2.waitKey(0)