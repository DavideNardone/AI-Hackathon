# New concepts and differences from Theano:
# - stride is the interval at which to apply the convolution
# - unlike previous course, we use constant-size input to the network
#   since not doing that caused us to start swapping
# - the output after convpool is a different size (8,8) here, (5,5) in Theano

# https://deeplearningcourses.com/c/deep-learning-convolutional-neural-networks-theano-tensorflow
# https://udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow
# from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import sys

import matplotlib.pyplot as plt
from matplotlib import pylab
import cv2
#import tensorflow as tf
# from FaceDetector import FaceDetector

#   import facenet libraires

from scipy import misc
# import tensorflow as tf
# import os
from align import detect_face

from pynput.keyboard import Key, Listener


def convpool(X, W, b):
    # just assume pool size is (2,2) because we need to augment it with 1s
    conv_out = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_out = tf.nn.bias_add(conv_out, b)
    pool_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.relu(pool_out)


def init_filter(shape, poolsz):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1] * np.prod(shape[:-2] / np.prod(poolsz)))
    return w.astype(np.float32)


show_key = False
# initial weights
M = 128
K = 7
poolsz = (2, 2)
dropout = 0.8

W1_shape = (3, 3, 1, 32)  # (filter_width, filter_height, num_color_channels, num_feature_maps)
W1_init = init_filter(W1_shape, poolsz)
b1_init = np.zeros(W1_shape[-1], dtype=np.float32)  # one bias per output feature map

W2_shape = (3, 3, 32, 64)  # (filter_width, filter_height, old_num_feature_maps, num_feature_maps)
W2_init = init_filter(W2_shape, poolsz)
b2_init = np.zeros(W2_shape[-1], dtype=np.float32)

W3_shape = (3, 3, 64, 128)  # (filter_width, filter_height, old_num_feature_maps, num_feature_maps)
W3_init = init_filter(W3_shape, poolsz)
b3_init = np.zeros(W3_shape[-1], dtype=np.float32)

# vanilla ANN weights
W4_init = np.random.randn(W3_shape[-1] * 6 * 6, M) / np.sqrt(W3_shape[-1] * 6 * 6 + M)
b4_init = np.zeros(M, dtype=np.float32)
W5_init = np.random.randn(M, M) / np.sqrt(M + M)
b5_init = np.zeros(M, dtype=np.float32)

W6_init = np.random.randn(M, K) / np.sqrt(M + K)
b6_init = np.zeros(K, dtype=np.float32)

# define variables and expressions
# using None as the first shape element takes up too much RAM unfortunately
X = tf.placeholder(tf.float32, shape=(None, 48, 48, 1), name='X')
T = tf.placeholder(tf.float32, shape=(None, K), name='T')
W1 = tf.Variable(W1_init.astype(np.float32))
b1 = tf.Variable(b1_init.astype(np.float32))
W2 = tf.Variable(W2_init.astype(np.float32))
b2 = tf.Variable(b2_init.astype(np.float32))
W3 = tf.Variable(W3_init.astype(np.float32))
b3 = tf.Variable(b3_init.astype(np.float32))
W4 = tf.Variable(W4_init.astype(np.float32))
b4 = tf.Variable(b4_init.astype(np.float32))
W5 = tf.Variable(W5_init.astype(np.float32))
b5 = tf.Variable(b5_init.astype(np.float32))
W6 = tf.Variable(W6_init.astype(np.float32))
b6 = tf.Variable(b6_init.astype(np.float32))

Z1 = convpool(X, W1, b1)
Z2 = convpool(Z1, W2, b2)
Z3 = convpool(Z2, W3, b3)

Z3_shape = Z3.get_shape().as_list()
# Z3r = tf.reshape(Z3, [Z3_shape[0], np.prod(Z3_shape[1:])])
Z3r = tf.reshape(Z3, [-1, np.prod(Z3_shape[1:])])

Z4 = tf.matmul(Z3r, W4) + b4
dropout_layer_d1 = tf.nn.relu(tf.nn.dropout(Z4, dropout))

Z5 = tf.matmul(dropout_layer_d1, W5) + b5
dropout_layer_d2 = tf.nn.sigmoid(tf.nn.dropout(Z5, dropout))

Yish = tf.matmul(dropout_layer_d2, W6) + b6

cost = tf.reduce_sum(
    tf.nn.softmax_cross_entropy_with_logits(
        logits=Yish,
        labels=T
    )
)

softmax_pred = tf.nn.softmax(
    logits=Yish
)

# we'll use this to calculate the error rate
predict_op = tf.argmax(Yish, 1)


def evaluate(session, face):
        Xeval = np.zeros([1, 48, 48, 1])
        Xeval[0, :, :, 0] = face[:]
        prediction = session.run(predict_op, feed_dict={X: Xeval})
        prediction_prob = session.run(softmax_pred, feed_dict={X: Xeval})
        print("Prediction: ", prediction)
        print("Prediction Prob: ", prediction_prob)
        return prediction, prediction_prob


def on_press(key):
    global show_key
    show_key = not show_key


def showAudienceAttentionBar(attention):

    plt.figure(2)
    print (attention)

    data = ((0, 0), (0, 0), (0, 0), attention, (0, 0), (0, 0), (0, 0))
    labels = ("", "", "", "", "", "", "")

    pylab.xlabel("Focus/No focus")
    pylab.title("Audience attention")
    pylab.gca().set_yscale('log')

    colors = ['green', 'red']

    x = np.arange(len(data))
    for i in xrange(len(data[0])):
        y = [d[i] for d in data]
        pylab.bar(x + i * 0.8, y, width=0.8, color=colors[i])
    pylab.gca().set_xticks(x)
    pylab.gca().set_xticklabels(labels)
    pylab.gca().set_ylim((0, 1000 + np.array(attention).max()))

    if show_key:
        pylab.draw()


def main():

    with Listener(on_press=on_press) as listener:
        # pass

        minsize = 30 # minimum size of face
        threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        factor = 0.709 # scale factor

        init = tf.global_variables_initializer()
        with tf.Session() as session:

            session.run(init)
            print("Restoring model")
            saver = tf.train.Saver()
            saver.restore(session, "/Users/davidenardone/PycharmProjects/AI-Hackathon/face_expression/my_test_model")
            print("Done!")

            # cap = cv2.VideoCapture('conf_video2.mp4')
            cap = cv2.VideoCapture(0)

            # detector = FaceDetector('haarcascade_frontalface.xml')
            category_faces = {
                0:  'Angry',
                1:  'Disgust',
                2:  'Fear',
                3:  'Happy',
                4:  'Sad',
                5:  'Surprise',
                6:  'Neutral'
            }

            focus = {
                0: 'No focus',
                1: 'No focus',
                2: 'Focus',
                3: 'Focus',
                4: 'No focus',
                5: 'Focus',
                6: 'Neutral'
            }

            focus_counter = {
                'No focus': .00001,
                'Focus': .00001
            }

            fig, ax = plt.subplots()
            fig2, _ = plt.subplots()

            y_pos = np.arange(len(category_faces.keys()))

            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel('Probability')
            ax.set_title('Facial expression')

            with session.as_default():
               pnet, rnet, onet = detect_face.create_mtcnn(session, None)

            count = 0
            focus_frame_counter = 0

            while True:
                ret, frame = cap.read()
                if frame is None:
                    print('None')
                else:
                    frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
                    pass
                # print (faces[:][:])

                faces, _ = detect_face.detect_face(
                        frame, minsize, pnet,
                        rnet, onet, threshold, factor)

                if focus_frame_counter > 7:
                    focus_frame_counter = 0
                    # if focus_counter['No focus'] > focus_counter['Focus']:

                    showAudienceAttentionBar((focus_counter['Focus'], focus_counter['No focus']))
                else:
                    focus_frame_counter += 1

                if count > 5:
                    ax.cla()
                    count = 0
                else:
                    count += 1

                for (x1, y1, x2, y2, acc) in faces:
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    w = x2-x1
                    h = y2-y1

                    #   plot the box using cv2

                    if w > h and y1 + w < frame.shape[0]:
                        h = w
                    elif w < h and y1 + h < frame.shape[1]:
                        w = h

                    cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (255, 0, 0), 2)

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # print(y1)
                    # print(y1+h)
                    # print(x1)
                    # print(x1+w)
                    # print(w)
                    # print(x2)
                    # print(frame.shape)
                    roi_gray = gray[y1:y1+h, x1:x1+w]
                    # print (roi_gray.shape)

                    if roi_gray.shape[0] == 0 or roi_gray.shape[1] == 0:
                        continue

                    # print (roi_gray.shape)

                    face = cv2.resize(roi_gray, (48, 48))
                    pred, pred_prob = evaluate(session, face)

                    if pred[0] == 6:
                        focus_counter['No focus'] += 0.5
                        focus_counter['Focus'] += 0.5
                    else:
                        focus_counter[focus[pred[0]]] += 1
                    # print (focus_counter)

                    plt.figure(1)

                    ax.barh(y_pos, pred_prob[0], align='center', color='green')

                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(category_faces.values())

                    if show_key:
                        plt.draw()
                        plt.pause(0.001)

                    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
                    cv2.putText(frame, category_faces[pred[0]], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

                if show_key:
                    cv2.imshow('frame',frame)
                    key = cv2.waitKey(42)
                    if (key == 27):
                        sys.exit(0)


if __name__ == '__main__':
    main()
