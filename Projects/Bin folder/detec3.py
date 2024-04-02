import cv2
import numpy as np
import math


def loose_equ(a,b, wiggle):
     return abs(a-b)<=wiggle

class ShapeDetector:
    def __init__(self):
        pass
    
    def unit_vector(self,v):#Функция для нахождения значения вектора
        return v/np.linalg.norm(v)

    def loose_equ(a,b, wiggle):
         return abs(a-b)<=wiggle
         

    def get_angle(self, p1, p2, p3):#Функция для нахождения угла
        v1 = np.array([p1[0]-p2[0],p1[1]-p2[1]])#находим вектор 1
        v2 = np.array([p1[0]-p3[0],p1[1]-p3[1]])#находим вектор 2
        v1_unit = self.unit_vector(v1)
        v2_unit = self.unit_vector(v2)
        radians = np.arccos(np.clip(np.dot(v1_unit,v2_unit),-1,1))
        return math.degrees(radians)

    def detect(self, c):
        # 3 ВЕРШИНЫ 
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        p1 = approx[0][0]#Присваиваем вершину 1
        p2 = approx[1][0]#Присваиваем вершину 2
        p3 = approx[2][0]#Присваиваем вершину 3

        if len(approx) == 3:
            degr1 = self.get_angle(p1,p2,p3)#Рассчитываем угол
            degr2 = self.get_angle(p2,p3,p1)
            degr3 = self.get_angle(p3,p1,p2)
            # print("triangle")
            # print(degr1)
            # print(degr2)
            # print(degr3)
            
            if (loose_equ(degr1,60,2) and loose_equ(degr2,60,2)):#Смотрим, если есть два угла в 60град
                shape = "equal triangle"#РАВНОСТОРОННИЙ ТРЕУГОЛЬНИК
            elif (loose_equ(degr1,degr2,3) or loose_equ(degr1,degr3,3) or loose_equ(degr3,degr2,3)):
                shape = "isosceles triangle"#РАВНОБЕДРЕННЫЙ ТРЕУГОЛЬНИК
            else:
                shape = "triangle"#ТРЕУГОЛЬНИК
           
           
            # 4 ВЕРШИНЫ
        elif len(approx) == 4:
            
            p4 = approx[3][0]
            degr1 = self.get_angle(p2,p1,p3)
            degr2 = self.get_angle(p1,p2,p4)
            degr4 = self.get_angle(p3,p4,p2)
            degr3 = self.get_angle(p4,p3,p1)
            pairing = [degr1 + degr2, degr1 + degr3, degr1 + degr4, degr2 + degr3, degr2 + degr4, degr3 + degr4]
            count=0
            # print("count={}".format(count))
            
            if ((pairing[0]>177)and(pairing[0]<183)):
                     count+=1
            if ((pairing[1]>177)and(pairing[1]<183)):
                     count+=1
            if ((pairing[2]>177)and(pairing[2]<183)):
                     count+=1
            if ((pairing[3]>177)and(pairing[3]<183)):
                     count+=1
            if ((pairing[4]>177)and(pairing[4]<183)):
                     count+=1
            if ((pairing[5]>177)and(pairing[5]<183)):
                     count+=1
            # print("COUNT = {}".format(count))

            if (loose_equ(degr1,90,2) and loose_equ(degr2,90,2) and loose_equ(degr3,90,2)):
                # compute the bounding box of the contour and use the
                # bounding box to compute the aspect ratio
                (x, y, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)
                shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"#либо КВАДРАТ, либо ПРЯМОУГОЛЬНИК
            elif (loose_equ(degr1,degr3,3) and loose_equ(degr2,degr4,3)):
                mid= (p1+p3)*.5
                degrm = self.get_angle(mid,p1,p2)

                # print("p1 = [{}][{}]".format(p1[0],p1[1]))
                # print("mid = [{}][{}]".format(mid[0],mid[1]))
                # print("p3 = [{}][{}]".format(p3[0],p3[1]))
                # print("mang = {}".format(degrm))

                if ((degrm>87)and(degrm<93)):
                    shape = "diamond"#РОМБ
                else: 
                    shape = "paral"#ПАРАЛЛЕЛОГРАМ
            elif (count==2):
                shape = "trapezoid"#ТРАПЕЦИЯ
            else:
                shape = "tetragon"
            # print(shape)
            # print(degr1)
            # print(degr2)
            # print(degr3)
            # print(degr4)
            
        elif len(approx) == 5: # 5 ВЕРШИН
            shape = "pentagon"
            
        elif len(approx) == 6: # 6 ВЕРШИН
            shape = "hexagon"
            
        elif len(approx) == 7: # 7 ВЕРШИН
            shape = "heptagon"
            
        else:# ОКРУЖНОСТЬ 8+
            shape = "circle"
            # return the name of the shape
        return shape