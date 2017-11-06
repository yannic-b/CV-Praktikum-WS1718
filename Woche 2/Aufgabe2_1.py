#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:28:32 2017

@author: Yannic
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

image1 = misc.imread('./LennaG.png')

plt.imshow(image1, cmap='Greys_r')

print 'Anzahl Pixel: ' + str(image1.shape[0]*image1.shape[1])

print 'Minimum: ' + str(image1.min())

print 'Maximum: ' + str(image1.max())

print 'Mittelwert: ' + str(image1.mean())

print 'Median: ' + str(np.median(image1))

print 'Standardabweichung: ' + str(np.std(image1))

def coordsForValue(input):
    out = []
    for (x,y), value in np.ndenumerate(image1):
        if input == value:
            out.append((x,y))
    return out

def numberOfValue(input):
    out = 0
    for (x,y), value in np.ndenumerate(image1):
        if input == value:
            out += 1
    return out

print 'Position min: ' + str(coordsForValue(0))
print 'Position max: ' + str(coordsForValue(255))

invertedImage1 = 255 - image1

#plt.imshow(invertedImage1, cmap='Greys_r')

mirrorImage1 = np.flip(image1, axis=1)

#plt.imshow(mirrorImage1, cmap='Greys_r')

partImage1 = image1[220:370, 220:350]

plt.imshow(partImage1, cmap='Greys_r')

brightImage1 = image1 + 100

#plt.imshow(brightImage1, cmap='Greys_r')

plt.show()

'''
x = 0
for i in xrange(255):
    x += numberOfValue(i)
print x
'''

def coordsForValueTest(value):
    index = np.argmax(image1, axis=None)
    return (index / 512, 512-(index/512))