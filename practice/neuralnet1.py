import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]=""
from keras.models import Sequential
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adagrad, Adam
from keras.utils import np_utils
from keras.preprocessing import image
import numpy as np
from PIL import Image
import random
import pandas as pd
#from image_descriptor import format_img, gen_features

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


#generate shuffled lists of integers to use as keys to create a shuffled dataframe
n = []
for i in range(0,50):
    n.append(i)
random.shuffle(n)

m = []
for i in range(50000, 50050):
    m.append(i)
random.shuffle(m)

#generate dataframes of train and test data, using random list to shuffle images along with their status (0 or 1)"""
PATH1 = "data/lightCurvePlots/trainData/"
PATH2 = "data/lightCurvePlots/testData/"
trainimages = pd.DataFrame(columns = ['image', 'status'])
trainimages = np.zeros((50,480,640,3))
trainstatus = np.zeros(50)
testimages = pd.DataFrame(columns=['image', 'status'])
testimages = np.zeros((50,480,640,3))
teststatus = np.zeros(50)
for i in range(0, 50, 2): #50000
    imgNum = n[i]
    string1 = PATH1 + "0fig" + str(imgNum) + ".png"
    string2 = PATH1 + "1fig" + str(imgNum) + ".png"
    arr1 = image.img_to_array(image.load_img(string1))
    arr1 = arr1.reshape((1,) + arr1.shape)
    arr2 = image.img_to_array(image.load_img(string2))
    arr2 = arr2.reshape((1,) + arr2.shape)
    trainimages[i] = arr1
    trainstatus[i] = 0
    trainimages[i+1] =arr2
    trainstatus[i+1] = 1
    trainimages[i] = [arr1, 0]
    trainimages[i+1] = [arr2, 1]

"""for i in range(0, 10000, 2):
    imgNum = m[i]
    string1 = PATH2 + "0fig" + str(imgNum) + ".png"
    string2 = PATH2 + "1fig" + str(imgNum) + ".png"
    img1 = image.load_img(string1)
    img2 = image.load_img(string2)
    arr1 = image.img_to_array(img1)
    arr1 = arr1.reshape((1,) + arr1.shape)
    arr2 = image.img_to_array(img2)
    arr2 = arr2.reshape((1,) + arr2.shape)
    testimages[i] = arr1
    teststatus[i] = 0
    testimages[i + 1] = arr2
    teststatus[i + 1] = 1
    testimages.loc[i] = [arr1, 0]
    testimages.loc[i+1] = [arr2, 1]"""

X = trainimages #480,640,3
Y = trainstatus



def attempt1():
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=(480,640,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    model.fit(X,Y, batch_size = 40, epochs = 10)




attempt1()


    model = Sequential()

    # Then add a first layer
    model.add(Dense(3000, input_shape=(100, 7)))
    # Define the activation function to use on the nodes of that first layer
    model.add(Activation('relu'))

    # Second hidden layer
    model.add(Dense(1000))
    model.add(Activation('relu'))

    # Third hidden layer
    model.add(Dense(500))
    model.add(Activation('relu'))

    # Fourth hidden layer
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Flatten())
    # Output layer with 10 categories (+using softmax)
    model.add(Dense(1))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.summary()
    adagrad = Adagrad(lr=1e-4)
    model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

    model.fit_generator(generator=keytraining_generator,
                    steps_per_epoch=len(keypartition['train']) // 64,
                    epochs=32,
                    validation_data=keyvalidation_generator,
                    validation_steps=len(keypartition['validation']) // 64)
    model.evaluate_generator(generator=keytraining_generator,
                         steps=len(keypartition['train']) // 64)