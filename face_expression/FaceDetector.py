import numpy as np
import cv2

class FaceDetector:

    def __init__(self,modelname):
        self.modelname = modelname
        self.cascade_classifier = cv2.CascadeClassifier(self.modelname)

    def detect_faces(self,img,show_image=False):

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        print(gray.shape)

        faces = self.cascade_classifier.detectMultiScale(gray,1.3,5)
        
        return faces, gray