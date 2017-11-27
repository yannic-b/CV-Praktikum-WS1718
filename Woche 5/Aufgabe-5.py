# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:28:07 2017

@author: 5kleber
"""

from glob import glob
from time import time
import numpy as np
from scipy import misc, ndimage
from skimage import filters, measure, feature
import matplotlib.pyplot as plt

# Aufgabe 1
nLenna = misc.imread('noisyLenna.png')
vec = np.array([[-1, -1], [-1, 0], [0, -1], [0, 0], [-1, 1], [1, -1], [0, 1], [1, 0], [1, 1]])

def denoise(img):
    denoiseLenna = img * 0
    for (x,y), value in np.ndenumerate(img):
        #print x, y
        sum = 0
        count = 0
        for v in vec:
            coords = np.array([x, y])
            if np.all((coords + v) >= 0) & np.all((coords + v) < img.shape):
                sum += img[x + v[0], y + v[1]]
                count += 1
        denoiseLenna[x, y] = int(sum/count)
    return denoiseLenna

def scipyDenoise(img):
    size = 9
    kernel = np.full((size,size), 1.0/(size*size))
    return ndimage.convolve(img, kernel, mode='mirror')

start = time()
#denoise(nLenna)
end = time()
print "manual denoise took", end-start, "seconds"

start = time()
scipyDenoise(nLenna)
end = time()
print "Scipy denoise took", end-start, "seconds"

#plt.imshow(scipyDenoise(nLenna), cmap='Greys_r')

# Aufgabe 3
lenna = misc.imread('Lenna.png')

def sobel(img):
    xSobel = filters.sobel_h(img)
    ySobel = filters.sobel_v(img)
    sobel = filters.sobel(img)
    gradientVec = np.stack((xSobel, ySobel), axis=-1)
    print gradientVec
    gradient = np.linalg.norm(gradientVec, axis=-1)
    
    fig = plt.figure()
    fig.add_subplot(1,4,1)
    plt.imshow(xSobel, cmap='Greys_r')
    fig.add_subplot(1,4,2)
    plt.imshow(ySobel, cmap='Greys_r')
    fig.add_subplot(1,4,3)
    plt.imshow(sobel, cmap='Greys_r')
    fig.add_subplot(1,4,4)
    plt.imshow(gradient, cmap='Greys_r')

#sobel(nLenna)    
#sobel(filters.gaussian(nLenna, 5))


# Aufgabe 5:
eye = misc.imread('auge.png')

def match(img, template):
    percentMatch = feature.match_template(img, template)
    highestMatch = np.argmax(percentMatch)
    argmaxCoords = np.unravel_index(highestMatch, percentMatch.shape)
    print argmaxCoords, percentMatch.shape
    plt.plot(argmaxCoords[1], argmaxCoords[0], 'bo')
    plt.imshow(img)

#match(lenna, eye)

whereis = misc.imread('whereIsWally1.jpg')
wally = misc.imread('wally.png')
#plt.imshow(whereis)
match(whereis, wally)
















