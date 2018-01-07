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
import os
import image_descriptor
import sqlite3


"""database_file = "lightcurve_data.sqlite"
table_name = 'lightcurve_data'
field1 = 'images'
field_type = 'BLOB'
conn = sqlite3.connect(database_file)
c = conn.cursor()
c.execute('CREATE TABLE {tn} ({nf} {ft})'\
          .format(tn=table_name, nf = field1, ft = field_type))
conn.commit()
conn.close()"""

field2 = 'status'
field_type2 =