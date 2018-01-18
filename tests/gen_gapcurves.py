"""Author Dahlia Dry
   Last Modified 12/8/2017
   This program generates raw data and png lightcurve images with periodic gaps
"""


from PIL import Image
import cv2
from matplotlib import pyplot as plt
import os
from image_descriptor import *
import numpy as np
import sys
import colormap as cm

script_dir = os.path.dirname(__file__)


def blackout(x0, y0, intblk, intshow):
    # x0, y0 = dims, intblk = width of blackout, intshow = width of observation period
    for i in range(0,5000): #5000
        rel_path0 = "data/lightCurvePlots/1fig%d.png" % i
        READ_PATH = os.path.join(script_dir, rel_path0)
        rel_path1 = "data/gapLightCurvePlots/1gapfig%d.png" % i
        OUT_PATH = os.path.join(script_dir, rel_path1)
        # source_img = Image.open(READ_PATH).convert("RGBA")
        img0 = format_img(READ_PATH)
        source_img = Image.fromarray(img0)
        blackout_img = Image.new('RGBA', (intblk, y0), "black")
        buffer = 0
        while buffer < x0:
            if (x0 - buffer) >= intblk:
                source_img.paste(blackout_img, (buffer, 0))
            elif (x0 - buffer) < intblk:

                blackout_img = Image.new('RGBA', ((x0 - buffer), y0))
                source_img.paste(blackout_img, (buffer, 0))
            source_img.save(OUT_PATH, "PNG")
            buffer = buffer + (intshow + intblk)

# blackout(490,360,40,40)


def gap_rawdata(status):
    for i in range(0,1000):
        loadfile = 'data/rawData/' + str(status) + 'raw%d.txt' % i
        savefile = 'data/gapData/' + str(status) + 'gapraw%d.txt' % i
        array = np.loadtxt(loadfile)
        lowval = sys.maxsize
        highval = -1e99
        for a in range(0, 360):
            if array[a] < lowval:
                lowval = array[a]
            if array[a] > highval:
                highval = array[a]
        r = highval - lowval
        for j in range(1,60):
            array[j] = lowval-(r/1000)
        for k in range(120,180):
            array[k] = lowval-(r/1000)
        for l in range(240,300):
            array[l] = lowval-(r/1000)
        np.savetxt(savefile, array)

gap_rawdata(0)


def test():
    s = np.loadtxt('data/gapData/0gapraw0.txt')
    t = np.loadtxt('data/rawData/0raw0.txt')
    plt.plot(t)
    plt.show()
    plt.plot(s)
    plt.show()

