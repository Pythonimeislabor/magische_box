import cv2
import numpy as np
import time

zahl = 9

bilder_gemacht = 0
cam = cv2.VideoCapture(0)
for i in range(10000):
    _, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.blur(frame,(25,25))
    frame = cv2.resize(frame,(28,28))
    frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 7);
    frame = np.max(frame) - frame
    cv2.imshow("",frame)
    key =  cv2.waitKey(30)
    
    
    if key == ord(" "):
        cv2.imwrite("unsere_daten/"+str(zahl)+"/"+str(zahl)+"_"+str(time.time())+".jpg", frame)
        bilder_gemacht+=1
        print(bilder_gemacht)
    
    if key == 27:
        break
        
    
    
