import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adagrad, Adam
from keras.utils import np_utils
from keras.preprocessing import image
from tools import threadsafe_generator
from DataGenerator import *
from keras.utils.np_utils import to_categorical

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



def genkeymodel():
    keypartition, keylabels, keylistIDs = gen_referenceKeyDicts()
    keytraining_generator = KeyDataGenerator(**keyparams).generate(keylabels, keypartition['train'])
    keyvalidation_generator = KeyDataGenerator(**keyparams).generate(keylabels, keypartition['validation'])

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



def genfigmodel1():
    figpartition, figlabels, figlistIDs = gen_referenceFigDicts()
    figtraining_generator = FigDataGenerator(**figparams).generate(figlabels, figpartition['train'])
    figvalidation_generator = FigDataGenerator(**figparams).generate(figlabels, figpartition['validation'])

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

def genkeymodel2():
    keypartition, keylabels, keylistIDs = gen_referenceKeyDicts()
    keytraining_generator = KeyDataGenerator(**keyparams).generate(keylabels, keypartition['train'])
    keyvalidation_generator = KeyDataGenerator(**keyparams).generate(keylabels, keypartition['validation'])

    vgg_model = Sequential()

    # On the very first layer, you must specify the input shape
    vgg_model.add(ZeroPadding2D((1, 1), input_shape=(100,7,1)))

    # Your first convolutional layer will have 64 3x3 filters,
    # and will use a relu activation function
    vgg_model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))

    # Once again you must add padding
    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))

    # Add a pooling layer with window size 2x2
    # The stride indicates the distance between each pooled window
    #vgg_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Lots more Convolutional and Pooling layers

    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    vgg_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    vgg_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Flatten the input
    vgg_model.add(Flatten())

    # Add a fully connected layer with 4096 neurons
    vgg_model.add(Dense(4096, activation='relu'))

    # Add a dropout layer
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


def genkeymodel3():
    keypartition, keylabels, keylistIDs = gen_referenceKeyDicts()
    keytraining_generator = KeyDataGenerator(**keyparams).generate(keylabels, keypartition['train'])
    keyvalidation_generator = KeyDataGenerator(**keyparams).generate(keylabels, keypartition['validation'])

    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu',
                     input_shape=(100,7,1)))
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


genkeymodel()

