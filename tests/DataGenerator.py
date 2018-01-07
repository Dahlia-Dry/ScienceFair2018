from image_descriptor import *
import keras
import os
import numpy as np
from tools import threadsafe_generator
from PIL import Image
import math

#dim  = 100
keydatasize = 5000
figdatasize = 5000

def test_read():
    script_dir = os.path.dirname(__file__)
    rel_path1 = "data/lightCurveKeypoints/0key0.txt"
    path = os.path.join(script_dir, rel_path1)
    kp = read_keypoints(path)
    img = format_img(os.path.join(script_dir, "data/lightCurvePlots/trainData/0fig0.png"))
    show_features(img, kp)

def gen_referenceKeyDicts():
    partition = {"train": [], "validation": []}
    labels = {}
    listIDs = []
    for i in range(0, keydatasize):
        id0 = "0key%d" % i
        id1 = "1key%d" % i
        listIDs.append(id0)
        listIDs.append(id1)
        if i % 4 != 0:
            partition["train"].append(id0)
            labels[id0] = 0
            partition["train"].append(id1)
            labels[id1] = 1
        else:
            partition["validation"].append(id0)
            labels[id0] = 0
            partition["validation"].append(id1)
            labels[id1] = 1
    return partition, labels, listIDs

def gen_referenceFigDicts():
    partition = {"train": [], "validation": []}
    labels = {}
    listIDs = []
    for i in range(0, figdatasize):
        id0 = "0fig%d" % i
        id1 = "1fig%d" % i
        listIDs.append(id0)
        listIDs.append(id1)
        if i % 4 != 0:
            partition["train"].append(id0)
            labels[id0] = 0
            partition["train"].append(id1)
            labels[id1] = 1
        else:
            partition["validation"].append(id0)
            labels[id0] = 0
            partition["validation"].append(id1)
            labels[id1] = 1
    return partition, labels, listIDs


class KeyDataGenerator(object):
    def __init__(self, dim_N = 100, dim_z = 7,
                 batch_size = 32, shuffle = True):
        #initialization
        self.dim_N = dim_N #162 keypoints in 1 set
        self.dim_z = dim_z #attributes list
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __get_exploration_order(self, listIDs):
        #generates order of exploration
        indexes = np.arange(len(listIDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)
        return indexes

    def __data_generation(self, labels, list_IDs_temp):
        script_dir = os.path.dirname(__file__)
        #Initialization
        X = np.empty((self.batch_size, self.dim_N, self.dim_z))
        y = np.empty((self.batch_size), dtype = int)
        #Generate Data
        for i, ID in enumerate(list_IDs_temp):
            path = os.path.join(script_dir, "data/lightCurveKeypoints/%s.txt" %ID)
            kp = read_keypoints(path)
            array = np.zeros([len(kp),7])
            for j in range(0, len(kp)):
                point_x = kp[j].pt[0]
                point_y = kp[j].pt[1]
                size = kp[j].size
                angle = kp[j].angle
                response = kp[j].response
                octave = kp[j].octave
                classid = kp[j].class_id

                array[j] = [point_x, point_y, size, angle, response, octave, classid]


            for a in range(0, len(array)):
                nans = np.isnan(array[a])
                for b in range(0, len(nans)):
                    if nans[b] == True:
                        array[a][b] = 0

            X[i,:] = array
            y[i] = labels[ID]
        return (X, y)

    @threadsafe_generator
    def generate(self, labels, listIDs):
        #Generates batches of samples
        #Infinite loop
        while 1:
            #Generate Order of Exploration
            indexes = self.__get_exploration_order(listIDs)

            #Generate batches
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                #Find list of IDs
                list_IDs_temp = [listIDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

                #Generate Data
                X,y = self.__data_generation(labels, list_IDs_temp)
                yield X,y


class FigDataGenerator(object):
    def __init__(self, dim_x = 480, dim_y = 640, dim_color = 2,
                 batch_size = 32, shuffle = True):
        #initialization
        self.dim_x = dim_x #640 pixel width
        self.dim_y = dim_y #480 pixel height
        self.dim_color = dim_color #2
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __get_exploration_order(self, listIDs):
        #generates order of exploration
        indexes = np.arange(len(listIDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)
        return indexes

    def __data_generation(self, labels, list_IDs_temp):
        script_dir = os.path.dirname(__file__)
        #Initialization
        X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_color))
        y = np.empty((self.batch_size), dtype = int)

        #Generate Data
        for i, ID in enumerate(list_IDs_temp):
            path = os.path.join(script_dir, "data/lightCurvePlots/trainData/%s.png" %ID)
            img = Image.open(path).convert('LA')
            X[i,:, :, :] = np.array(img)

            y[i] = labels[ID]
        return (X, y)

    @threadsafe_generator
    def generate(self, labels, listIDs):
        #Generates batches of samples
        #Infinite loop
        while 1:
            #Generate Order of Exploration
            indexes = self.__get_exploration_order(listIDs)

            #Generate batches
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                #Find list of IDs
                list_IDs_temp = [listIDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

                #Generate Data
                X,y = self.__data_generation(labels, list_IDs_temp)

                yield X,y

