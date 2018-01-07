import cv2
from matplotlib import pyplot as plt
import _pickle
import operator
import math


def format_img(imgname):
    img = cv2.imread(imgname, 0)
    img = img[60:420, 80:570]
    return img

def gen_features(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(img, None)
    dictkps = {}
    newkp = []
    for i in range(0, len(kp)):
        dictkps['%d' % i] = kp[i].response
    sorted_dictkps = sorted(dictkps.items(), key = operator.itemgetter(1))

    r = len(sorted_dictkps) - 100

    indexes = []
    for j in range(r, len(sorted_dictkps)):
        indexes.append(sorted_dictkps[j][0])

    for k in indexes:
        newkp.append(kp[int(k)])
    return (newkp, desc)

def gen_alt_keypoints(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img)
    return kp

def show_features(img, kp):
    res = cv2.drawKeypoints(img, kp, img.copy())
    plt.imshow(res)
    plt.show()

def map_keypoint_matches(img1, img2, kp1, kp2, desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key = lambda x:x.distance)
    N_MATCHES = 100
    match_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches[:N_MATCHES], img2.copy(), flags=0
    )
    return match_img

def write_keypoints(kp, file_path): #write keypoints to binary file
    index = []
    for p in kp:
        temp = (p[0], p[1], p[2], p[3], p[4], p[5], p[6])
        index.append(temp)

    w= open(file_path, "wb")
    w.write(_pickle.dumps(index))
    w.close()

def read_keypoints(file_path): #return keypoint var compatible with cv2 methods
    r = open(file_path, "rb")
    rr = _pickle.load(r)
    kp = []
    for p in rr:
        #temp = cv2.KeyPoint(x=p[0][0], y=p[0][1], _size=p[1],
                            #_angle=p[2], _response = p[3],
                            #_octave=p[4],_class_id=p[5])
        temp = cv2.KeyPoint(x = p[0], y = p[1], _size = p[2],
                            _angle=p[3], _response= p[4],
                            _octave = p[5], _class_id = p[6])
        kp.append(temp)
    return kp

#to show image with mapped keypoints:
    # format_img() -> gen_features() -> show_features()

#to show 2 concatenated images with mapped keypoint matches:
    # format_img() -> gen_features() -> map_keypoint_matches()

#img = format_img('0fig0.png')
#(kp, desc) = gen_features(img)
