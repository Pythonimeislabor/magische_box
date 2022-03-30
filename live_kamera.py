import cv2
import numpy as np

kamera = cv2.VideoCapture(0)

for _ in range(2000):
    _, bild = kamera.read()
    
    bild = bild / 255
    filter_3x3 = [
                    [-1,-2,-1],
                    [-0,-0,-0],
                    [+1,+2,+1],
    ]
    filter_3x3 = np.array(filter_3x3)
    neues_bild1 = cv2.filter2D(bild, -1, filter_3x3)
    
    filter_3x3 = [
                    [-1,-0,+1],
                    [-2,-0,+2],
                    [-1,+0,+1],
    ]
    filter_3x3 = np.array(filter_3x3)
    neues_bild2 = cv2.filter2D(bild, -1, filter_3x3)
    
    neues_bild_summe = (neues_bild2 + neues_bild1) / 2
    
    filter_altes_bild = (neues_bild_summe * 3 + bild * 1) / 4
    
    cv2.imshow("filter_3x3", filter_altes_bild)
    cv2.waitKey(30)
    
kamera.release()