import cv2
import numpy as np
import imutils
####### training part ###############
samples = np.loadtxt('generalsamples.data', np.float32)
responses = np.loadtxt('generalresponses.data', np.float32)
responses = responses.reshape((responses.size, 1))
model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)
############################# testing part #########################
colors = {
'blue': (np.array([115, 160, 140]), np.array([130, 255,250])),
'green': (np.array([55, 170, 150]), np.array([72, 200, 180]),np.array([25, 160, 140]), np.array([40, 255, 250])),
'brown': (np.array([0, 160, 140]), np.array([3, 175, 170]),np.array([0, 0, 45]), np.array([1, 1, 53])),
'purple': (np.array([145, 165, 145]), np.array([170, 210, 251]))
}
relatives = {
    1: ('heart'),
    2: ('4pt Star'),
    3: ('arrow'),
    4: ('hex'),
    5: ('5pt Star'),
    #5:('blank')
}   


image = cv2.imread('figu.jpg')

im = imutils. resize(image, width= 600 )

# colorReduce()
div = 100
quant = im // div * div + div // 2
#cv2.imshow('quant', quant)

im2 = im.copy()
height, width, channels = im2.shape
fin = np.full((height, width, 3), 255, dtype=np.uint8) # create
#cv2.floodFill(fin, ())
def color_check():
    # blurred = cv2.GaussianBlur(quant, (5, 5), 0)
    hsv = cv2.cvtColor(quant, cv2.COLOR_BGR2HSV)
    #cv2.imshow('hsv', hsv)
    for col in colors:
        mask = cv2.inRange(hsv, colors.get(col)[0], colors.get(col)[1])
        if col == 'green':
            mask += cv2.inRange(hsv, colors.get(col)[2], colors.get(col)[3])
            #cv2.imshow('greentest', mask)
        # if col == 'blue':
        #     cv2.imshow('bluetest', mask)
        if col == 'brown':
            mask += cv2.inRange(hsv, colors.get(col)[2], colors.get(col)[3])
            #cv2.imshow('browntest', mask)
        # if col == 'purple':
        #     cv2.imshow('purpletest', mask)
        contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2:]
        contours = [i for i in contours if
        cv2.contourArea(i) > 50]
        fullcop(fin, contours, mask,col)
        find_fig(fin,contours,mask,col)

def fullcop(im,contours,mask,col):
    color = (0, 190, 255)
    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        if (col == 'blue'):
            color = (170, 0, 0)
        elif (col == 'brown'):
            color = (0, 70, 150)
        elif (col == 'green'):
            color = (0, 200, 50)
        elif (col == 'purple'):
            color = (150, 0, 150)
        cv2.drawContours(im, [cnt], -1, color, -1, cv2.LINE_AA)

def find_fig(im,contours,mask,col):
    color = (0, 190, 255)
    if (col == 'blue'):
        color = (170, 0, 0)
    elif (col == 'brown'):
        color = (0, 70, 150)
    elif (col == 'green'):
        color = (0, 200, 50)
    elif (col == 'purple'):
        color = (150, 0, 150)
    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        if h > 28:
            try:
                roi = mask[y:y + h, x:x + w]
                l = float(w) / h
                roismall = cv2.resize(roi, (10, 10))
                roismall = roismall.reshape((1, 100))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)
                num = int(results[0]) # ?
                result = 0
                result = relatives[num] ##ломается
                text = "{} {}".format(col, result)
                cv2.putText(im, text, (x + w // 2, y + h // 2), 0, 0.5, (0, 0, 0),3,cv2.LINE_AA)
                cv2.putText(im, text, (x + w // 2, y + h // 2), 0, 0.5, (color[0]+60,color[1]+60,color[2]+60),1,cv2.LINE_AA)
            except cv2.Error as e:
                print('Invalid')
color_check()
cv2.imshow('start', im)
#cv2.imshow('out', im2)
cv2.imshow('finale', fin)
cv2.waitKey(0)