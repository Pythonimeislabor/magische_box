import cv2
import tensorflow as tf
import numpy as np
import pigpio


pi = pigpio.pi()
pi.set_servo_pulsewidth(14,2500)
                        

model = tf.keras.models.load_model("model.h5")
model.summary()
kamera = cv2.VideoCapture(0)


def interpoliere(zahl):
    reichweite = 2000
    schrittweite = reichweite / 4
    return schrittweite * zahl + 500
    
    

letzte_ergebnisse = []

                        
for _ in range(10000):
    _, frame = kamera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.blur(frame,(25,25))
    frame = cv2.resize(frame,(28,28))
    frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 7);
    frame = np.max(frame) - frame
    frame = frame / np.max(frame)
    cv2.imshow("",frame)
    key =  cv2.waitKey(30)
    if key == 27:
        break

    
    batch = np.reshape(frame, (1,28,28))
    ergebnis = model(batch)
    
    # wackeln verhindern
    letzte_ergebnisse.append(ergebnis[0])
    if len(letzte_ergebnisse) == 5:
        del letzte_ergebnisse[0]
    
    xn0 = [x.numpy()[0] for x in letzte_ergebnisse]
    xn1 = [x.numpy()[1] for x in letzte_ergebnisse]
    xn2 = [x.numpy()[2] for x in letzte_ergebnisse]
    xn3 = [x.numpy()[3] for x in letzte_ergebnisse]
    
    durchschnitt = [
                    np.mean(xn0),
                    np.mean(xn1),
                    np.mean(xn2),
                    np.mean(xn3),
    ]
    
    zahl = np.argmax(durchschnitt)
    print("\r",zahl,end="  ")
    
    pi.set_servo_pulsewidth(14,interpoliere(zahl))

kamera.release()
