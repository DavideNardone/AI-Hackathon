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

import numpy as np
import tensorflow as tf
import os
import sys

import cv2
# import tensorflow as tf
from FaceDetector import FaceDetector


def convpool(X, W, b):
    # just assume pool size is (2,2) because we need to augment it with 1s
    conv_out = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_out = tf.nn.bias_add(conv_out, b)
    pool_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.relu(pool_out)


def init_filter(shape, poolsz):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1] * np.prod(shape[:-2] / np.prod(poolsz)))
    return w.astype(np.float32)


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


def main():
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        print("Restoring model")
        saver = tf.train.Saver()
        saver.restore(session, "my_test_model")
        print("Done!")

        # cap = cv2.VideoCapture('happiness.mov')
        cap = cv2.VideoCapture(0)
        detector = FaceDetector('haarcascade_frontalface.xml')
        category_faces = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
        while True:
            ret, frame = cap.read()
            if frame is None:
                print('None')
            else:
                frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
                faces, gray_frame = detector.detect_faces(frame, show_image=True)
            # print (faces[:][:])
            for (x, y, w, h) in faces:
                roi_gray = gray_frame[y:y + h, x:x + w]
                face = cv2.resize(roi_gray, (48, 48))
                pred, pred_prob = evaluate(session, face)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, category_faces[pred[0]], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))


                # f = np.array()

            cv2.imshow('frame', frame)
            key = cv2.waitKey(0)
            if (key == 27):
                sys.exit(0)


if __name__ == '__main__':
    main()
