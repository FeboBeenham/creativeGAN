

import os
import matplotlib.pyplot as plt
import cv2
import random

root ='./wikiart'

#Set your own PATH
PATH = os.path.normpath('BASEPATH/pr_')

for subdir, dirs, files in os.walk(root):
    style = subdir[2:]
    name = style
    if len(style) < 1:
        continue
    try:
        os.stat(PATH + name)
    except:
        os.mkdir(PATH + name)

    i = 0
    for f in files:
        source = style + '\\' + f
        print(str(i) + source)
        try:
            image = plt.imread(source)
            image = cv2.resize(image,(128, 128))
            plt.imsave(PATH + name + '\\' + str(i) + '.jpg',image)
            i += 1
        except Exception:
            print('missed it: ' + source)

