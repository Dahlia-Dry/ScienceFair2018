from image_descriptor import *
import os
import math
import numpy as np
script_dir = os.path.dirname(__file__)
def gen_keypoints(script_dir):
    for i in range (0, 5000):
        rel_path0 = "data/lightCurvePlots/0fig%d.png" % i
        READ_PATH = os.path.join(script_dir, rel_path0)
        img = format_img(READ_PATH)
        (kp, desc) = gen_features(img)
        for j in range(0, len(kp)):
            point_x = kp[j].pt[0]
            point_y = kp[j].pt[1]
            size = kp[j].size
            angle = kp[j].angle
            response = kp[j].response
            octave = kp[j].octave
            classid = kp[j].class_id
            if math.isnan(point_x):
                point_x = 0
            if math.isnan(point_y):
                point_y = 0
            if math.isnan(size):
                size = 0
            if math.isnan(angle):
                angle = 0
            if math.isnan(response):
                response = 0
            if math.isnan(octave):
                octave = 0
            if math.isnan(classid):
                classid = 0
            kp[j] = [point_x, point_y, size, angle, response, octave, classid]
            for k in range(0, len(kp[j])):
                if math.isnan(kp[j][k]):
                    print("yikes at %d %d" %(i,j))
        rel_path1 = "data/lightCurveKeypoints/0key%d.txt" % i
        WRITE_PATH = os.path.join(script_dir, rel_path1)
        write_keypoints(kp, WRITE_PATH)

gen_keypoints(script_dir)

def test():
    script_dir = os.path.dirname(__file__)
    rel_path1 = "data/lightCurveKeypoints/0key0.txt"
    path = os.path.join(script_dir, rel_path1)
    kp = read_keypoints(path)
    img = format_img(os.path.join(script_dir, "data/lightCurvePlots/0fig0.png"))


