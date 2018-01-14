# Author Dahlia Dry
# Last Modified 12/8/2017
# This program is the final version of the neural net used in this project.
# FigNeuralNet -> model2 is the neural net used, a CNN with 96% classifying accuracy

import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.layers.convolutional import ZeroPadding2D, ZeroPadding1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adagrad, Adam
from keras.utils import np_utils
from keras.preprocessing import image
from tools import threadsafe_generator
from DataGenerator import *
from keras.utils.np_utils import to_categorical
import os
from gen_keypoints import *
from image_descriptor import *

TF_CPP_MIN_LOG_LEVEL = 2

keyparams = {'dim_N': 100,
          'dim_z': 7,
          'batch_size': 32,
          'shuffle': True}

figparams = {'dim_x': 480,
             'dim_y': 640,
             'dim_color' : 2,
             'batch_size': 8,
             'shuffle': True}

#model = Sequential()
#model.add(Dense(32, input_shape=(100,7)))
#model.add(Flatten())
#model.add(Dense(1, activation = 'sigmoid'))


class KeyNeuralNet(KeyDataGenerator):
    def __init__(self, keyparams):
        KeyDataGenerator.__init__(self, **keyparams)

    def gen_model1(self): # MLP, 62% accuracy
        keypartition, keylabels, keylistIDs = self.gen_referenceKeyDicts()
        keytraining_generator = self.generate(keylabels, keypartition['train'])
        keyvalidation_generator = self.generate(keylabels, keypartition['validation'])

        model = Sequential()
        model.add(Dense(32, input_shape=(100,7)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(500, activation = 'relu'))
        model.add(Dense(2, activation = 'softmax'))
        model.add(Dense(1, activation = 'relu'))

        adam = Adam(clipnorm = 1)
        model.summary()
        adagrad = Adagrad(lr=1e-4)
        model.compile(loss='binary_crossentropy',
                    optimizer=adam,
                    metrics = ['accuracy'])

        model.fit_generator(generator=keytraining_generator,
                            steps_per_epoch=len(keypartition['train']) // 64,
                            epochs=32,
                            validation_data=keyvalidation_generator,
                            validation_steps=len(keypartition['validation']) // 64)
        model.evaluate_generator(generator=keytraining_generator,
                                steps=len(keypartition['train']) // 64)

    def gen_model2(self): # CNN, 96% accuracy
        keypartition, keylabels, keylistIDs = self.gen_referenceKeyDicts()
        keytraining_generator = self.generate(keylabels, keypartition['train'])
        keyvalidation_generator = self.generate(keylabels, keypartition['validation'])

        vgg_model = Sequential()

        vgg_model.add(ZeroPadding2D((1, 1), input_shape=(100, 7, 1)))

        vgg_model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))

        vgg_model.add(ZeroPadding2D((1, 1)))
        vgg_model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))

        vgg_model.add(ZeroPadding2D((1, 1)))
        vgg_model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
        vgg_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        vgg_model.add(ZeroPadding2D((1, 1)))
        vgg_model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
        vgg_model.add(ZeroPadding2D((1, 1)))
        vgg_model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
        vgg_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        vgg_model.add(Flatten())

        # fully connected layer with 4096 neurons
        vgg_model.add(Dense(4096, activation='relu'))

        # dropout layer
        vgg_model.add(Dropout(0.5))

        vgg_model.add(Dense(4096, activation='relu'))
        vgg_model.add(Dropout(0.5))

        vgg_model.add(Dense(1, activation='softmax'))

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        vgg_model.summary()
        adagrad = Adagrad(lr=1e-4)
        vgg_model.compile(loss='binary_crossentropy',
                          optimizer=sgd,
                          metrics=['accuracy'])

        vgg_model.fit_generator(generator=keytraining_generator,
                                steps_per_epoch=len(keypartition['train']) // 64,
                                epochs=32,
                                validation_data=keyvalidation_generator,
                                validation_steps=len(keypartition['validation']) // 64)
        vgg_model.evaluate_generator(generator=keytraining_generator,
                                     steps=len(keypartition['train']) // 64)

    def gen_model3(self):
        keypartition, keylabels, keylistIDs = self.gen_referenceKeyDicts()
        keytraining_generator = self.generate(keylabels, keypartition['train'])
        keyvalidation_generator = self.generate(keylabels, keypartition['validation'])

        model = Sequential()
        model.add(Convolution2D(32, kernel_size=(3, 3), strides=(1, 1),
                                activation='relu',
                                input_shape=(100, 7, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='binary_crossentropy',
                      optimizer=keras.optimizers.SGD(lr=0.01),
                      metrics=['accuracy'])
        model.fit_generator(generator=keytraining_generator,
                            steps_per_epoch=len(keypartition['train']) // 64,
                            epochs=32,
                            validation_data=keyvalidation_generator,
                            validation_steps=len(keypartition['validation']) // 64)

    def _gen_testing_data(self):
        script_dir = os.path.dirname(__file__)
        x_train = np.empty((2000, 100, 7))
        y_train = np.empty((2000))
        x_test = np.empty((2000, 100, 7))
        y_test = np.empty((2000))
        for i in range(0,2000, 2):
            path0 = os.path.join(script_dir, "data/lightCurveKeypoints/0key%d.txt" %i)
            path1 = os.path.join(script_dir, "data/lightCurveKeypoints/1key%d.txt" %i)
            y_train[i] = 0
            y_train[i+1] = 1
            kp0 = read_keypoints(path0)
            array0 = np.zeros([len(kp0), 7])
            for j in range(0, len(kp0)):
                point_x = kp0[j].pt[0]
                point_y = kp0[j].pt[1]
                size = kp0[j].size
                angle = kp0[j].angle
                response = kp0[j].response
                octave = kp0[j].octave
                classid = kp0[j].class_id
                array0[j] = [point_x, point_y, size, angle, response, octave, classid]
            kp1 = read_keypoints(path1)
            array1 = np.zeros([len(kp1), 7])
            for k in range(0, len(kp1)):
                point_x = kp1[k].pt[0]
                point_y = kp1[k].pt[1]
                size = kp1[k].size
                angle = kp1[k].angle
                response = kp1[k].response
                octave = kp1[k].octave
                classid = kp1[k].class_id
                array1[k] = [point_x, point_y, size, angle, response, octave, classid]
            for l in range(100):
                for m in range(7):
                    x_train[i][l][m] = array0[l][m]
                    x_train[i + 1][l][m] = array1[l][m]

        for i in range(2000, 4000, 2):
            path0 = os.path.join(script_dir, "data/lightCurveKeypoints/0key%d.txt" % i)
            path1 = os.path.join(script_dir, "data/lightCurveKeypoints/1key%d.txt" % i)
            y_test[i-2000] = 0
            y_test[i + 1 - 2000] = 1
            kp0 = read_keypoints(path0)
            array0 = np.zeros([len(kp0), 7])
            for j in range(0, len(kp0)):
                point_x = kp0[j].pt[0]
                point_y = kp0[j].pt[1]
                size = kp0[j].size
                angle = kp0[j].angle
                response = kp0[j].response
                octave = kp0[j].octave
                classid = kp0[j].class_id
                array0[j] = [point_x, point_y, size, angle, response, octave, classid]
            kp1 = read_keypoints(path1)
            array1 = np.zeros([len(kp1), 7])
            for k in range(0, len(kp1)):
                point_x = kp1[k].pt[0]
                point_y = kp1[k].pt[1]
                size = kp1[k].size
                angle = kp1[k].angle
                response = kp1[k].response
                octave = kp1[k].octave
                classid = kp1[k].class_id
                array1[k] = [point_x, point_y, size, angle, response, octave, classid]

            for l in range(100):
                for m in range(7):
                    x_train[i-2000][l][m] = array0[l][m]
                    x_train[i + 1-2000][l][m] = array1[l][m]

        return x_train, y_train, x_test, y_test

    def gen_model4(self):
        # keypartition, keylabels, keylistIDs = self.gen_referenceKeyDicts()
        # keytraining_generator = self.generate(keylabels, keypartition['train'])
        # keyvalidation_generator = self.generate(keylabels, keypartition['validation'])
        x_train, y_train, x_test, y_test = self._gen_testing_data()

        model = Sequential()
        model.add(ZeroPadding1D((1), input_shape=(100, 7)))
        model.add(Convolution1D(64, 3, activation='relu'))
        model.add(MaxPooling1D((2)))

        model.add(ZeroPadding1D((1)))
        model.add(Convolution1D(128, 3, activation='relu'))
        model.add(MaxPooling1D((2)))

        model.add(ZeroPadding1D((1)))
        model.add(Convolution1D(192, 3, activation='relu'))
        model.add(MaxPooling1D((2)))

        model.add(ZeroPadding1D((1)))
        model.add(Convolution1D(256, 3, activation='relu'))
        model.add(MaxPooling1D((2)))

        model.add(ZeroPadding1D((1)))
        model.add(Convolution1D(512, 3, activation='relu'))
        model.add(MaxPooling1D((2)))

        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)

        model.fit(x_train, y_train, batch_size = 100, nb_epoch=10)
        model.evaluate(x_test, y_test)
        """ model.fit_generator(generator=keytraining_generator,
                            steps_per_epoch=len(keypartition['train']) // 64,
                            epochs=32,
                            validation_data=keyvalidation_generator,
                            validation_steps=len(keypartition['validation']) // 64) """

        """ model.evaluate_generator(generator=keytraining_generator,
                                 steps=len(keypartition['train']) // 64) """

class FigNeuralNet(FigDataGenerator):
    def __init__(self, figparams):
        FigDataGenerator.__init__(self, **figparams)

    def gen_model1(self):
        figpartition, figlabels, figlistIDs = self.gen_referenceFigDicts()
        figtraining_generator = self.generate(figlabels, figpartition['train'])
        figvalidation_generator = self.generate(figlabels, figpartition['validation'])

        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(480,640,2)))
        model.add(Convolution2D(32, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(192, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='softmax'))

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.summary()
        adagrad = Adagrad(lr=1e-4)
        model.compile(loss='binary_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        model.fit_generator(generator=figtraining_generator,
                            steps_per_epoch=len(figpartition['train']) // 64,
                            epochs=32,
                            validation_data=figvalidation_generator,
                            validation_steps=len(figpartition['validation']) // 64)
        model.evaluate_generator(generator=figtraining_generator,
                                 steps=len(figpartition['train']) // 64)






nn = KeyNeuralNet(keyparams)
nn.gen_model4()

