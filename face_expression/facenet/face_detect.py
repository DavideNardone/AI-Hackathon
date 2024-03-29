#   import facenet libraires
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import os
import align.detect_face

#  import other libraries
import cv2

#   setup facenet parameters
gpu_memory_fraction = 1.0
minsize = 50 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

print ('voli dormire')

#   fetch images
# image_dir = 'presidents/'

#   create a list of your images
# images = os.listdir(image_dir)

cap = cv2.VideoCapture(0)

#   Start code from facenet/src/compare.py
print('Creating networks and loading parameters')
with tf.Graph().as_default():
    # gpu_options = tf.GPUOptions(
    #     per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(
        # gpu_options=gpu_options,
        log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
#   end code from facenet/src/compare.py  

    while True:
            ret, frame = cap.read()
            # img = misc.imread(os.path.expanduser(image_dir + i))
            #   run detect_face from the facenet library
            bounding_boxes, _ = align.detect_face.detect_face(
                    frame, minsize, pnet,
                    rnet, onet, threshold, factor)

            #   for each box
            for (x1, y1, x2, y2, acc) in bounding_boxes:
                w = x2-x1
                h = y2-y1
                #   plot the box using cv2
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x1+w),
                    int(y1+h)),(255,0,0),2)
                print ('Accuracy score', acc)
            #   save a new file with the boxed face
            # cv2.imwrite('faceBoxed', frame)
            #   show the boxed face
                cv2.imshow('facenet is cool', frame)
                print('Press any key to advance to the next image')
                cv2.waitKey(1)
