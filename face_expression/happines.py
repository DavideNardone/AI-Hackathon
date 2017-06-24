import cv2
#import tensorflow as tf
from FaceDetector import FaceDetector
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('happiness.mov')
detector = FaceDetector('haarcascade_frontalface.xml')

while True:
    ret,frame = cap.read()
    if frame is None:
        print('None')
    else:
        faces = detector.detect_faces(frame, show_image=True)
    print (faces[:][:])
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = frame[y:y + h, x:x + w]
    cv2.imshow('frame',frame)
    cv2.waitKey(41)