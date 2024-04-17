import cv2
import numpy as np
import imutils

samples = np.loadtxt('generalsamples.data', np.float32)
responses = np.loadtxt('generalresponses.data', np.float32)
responses = responses.reshape((responses.size, 1))
model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

shift_p = 5
shift_b = 21
shift_o = 10
shift_r = 5

colors = {
    'pink': (np.array([145 - shift_p, 145 - shift_p, 245 - shift_p]), np.array([162 + shift_p, 155 + shift_p, 255 + shift_p])),
    'orange': (np.array([17 - shift_o, 252 - shift_o, 235 - shift_o]), np.array([17 + shift_o, 255 + shift_o, 255 + shift_o])),
    'red': (np.array([1 - shift_r, 190 - shift_r, 250 - shift_r]), np.array([6 + shift_r, 255 + shift_r, 254 + shift_r])),
    'blue': (np.array([111 - shift_b, 164 - shift_b, 243 - shift_b]), np.array([116 + shift_b, 236 + shift_b, 254 + shift_b]))
}

shapes = {
    1: ('circle'),
    2: ('star'),
    3: ('gato'),
    4: ('tringle'),
    5: ('square')
}

image = cv2.imread('example.jpg')
base = imutils. resize(image, width=600)

im2 = base.copy()
height, width, channels = im2.shape
output = np.full((height, width, 3), 255, dtype=np.uint8)


def color_check():
    hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
    for col in colors:
        mask = cv2.inRange(hsv, colors.get(col)[0], colors.get(col)[1])

        """
        if col == 'pink':
            cv2.imshow('pinktest', mask)
        """

        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2:]

        contours = [i for i in contours if cv2.contourArea(i) > 50]
        fig_copy(output, contours, col)
        fig_find(output, contours, mask, col)


colours = {
    "Red": (0, 0, 255),
    "Orange": (0, 100, 255),
    "Blue": (255, 0, 0),
    "Pink": (255, 0, 242)
}


def fig_copy(im, contours, col):
    color = (0, 0, 0)
    for cnt in contours:
        match col:
            case 'red':
                color = colours["Red"]
            case 'blue':
                color = colours["Blue"]
            case 'orange':
                color = colours["Orange"]
            case 'pink':
                color = colours["Pink"]
            case _:
                color = "Color not found"
        cv2.drawContours(im, [cnt], -1, color, -1, cv2.LINE_AA)


def fig_find(im, contours, mask, col):
    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        if h > 28:
            try:
                roi = mask[y:y + h, x:x + w]
                roismall = cv2.resize(roi, (10, 10))
                roismall = roismall.reshape((1, 100))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)
                num = int(results[0])
                result = 0
                result = shapes[num]
                text = "{} {}".format(col, result)
                cv2.putText(im, text, (x + w // 2, y + h // 2), 1, 1, (150, 150, 150), 1, cv2.LINE_AA)
            except cv2.Error as e:
                print('Invalid')


color_check()
cv2.imshow('start', base)
cv2.imshow('finale', output)
cv2.waitKey(0)
