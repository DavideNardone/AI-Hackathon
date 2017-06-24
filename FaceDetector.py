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
        for (x, y, w, h) in faces:
            cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]

        if show_image == True:
            cv2.imshow("faces_grayscale",gray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return faces