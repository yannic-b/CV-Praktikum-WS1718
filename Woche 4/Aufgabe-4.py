# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:50:27 2017

@author: 5kleber
"""

import math
from glob import glob
import numpy as np
from scipy import misc, ndimage
from skimage import filters, measure
import matplotlib.pyplot as plt



def importData(number):
    
    hariTrain = glob('./haribo'+number+'/hariboTrain/*.png')

    trLabels = []
    for string in hariTrain:
        trLabels.append(string.split('/')[-1].split('.')[0].split('_')[0])
    
    trImages = []
    for img in hariTrain:
        trImages.append(misc.imread(img))
    
    hariVal = glob('./haribo'+number+'/hariboVal/*.png')
    
    valLabels = []
    for string in hariVal:
        valLabels.append(string.split('/')[-1].split('.')[0].split('_')[0])
    
    valImages = []
    for img in hariVal:
        valImages.append(misc.imread(img))
        
    return trLabels, trImages, valLabels, valImages
    
    
def l2Distance(img1, img2):
    return np.sqrt(np.square(img1[0] - img2[0]) + np.square(img1[1] - img2[1]) + np.square(img1[2] - img2[2]))
    
def calcMean(imgs):
    means = []
    for img in imgs:
        means.append(np.mean(img, axis=(0,1)))
    return np.array(means)

def meanClassify(trImgs, valImgs, trLabels, valLabels):
    trMeans = calcMean(trImgs)
    valMeans = calcMean(valImgs)
    testLabels = np.empty(valMeans.shape[0], dtype='object')
    for i in range(valMeans.shape[0]):
        minDistance = 256
        for j in range(trMeans.shape[0]):
            if l2Distance(valMeans[i], trMeans[j]) < minDistance:
                minDistance = l2Distance(valMeans[i], trMeans[j])
                testLabels[i] = trLabels[j]
    print np.array(list(zip(testLabels, valLabels)))
    correct = 0
    for x in range(testLabels.size):
        if testLabels[x] == valLabels[x]:
            correct += 1
    print correct, "von", len(valLabels), "(", correct/float(len(valLabels)), "% )"    


# Aufgabe 2
def boxImages(imgs):
    boxes = []
    for img in imgs:
        grey = np.dot(img[...,:3], [0.299, 0.587, 0.114])
        otsu = filters.threshold_otsu(grey) #+ 35
        binary = (grey <= otsu)
        props = measure.regionprops(binary.astype(int))[0]
        #box = props.bbox
        print math.degrees(props.orientation)
        rotImg = ndimage.rotate(img, angle=math.degrees(props.orientation), mode='nearest')
        outGrey = np.dot(rotImg[...,:3], [0.299, 0.587, 0.114])
        otsu2 = filters.threshold_otsu(outGrey) + 35
        outBinary = (outGrey <= otsu2)
        outProps = measure.regionprops(outBinary.astype(int))[0]
        outBox = outProps.bbox
        boxes.append(rotImg[outBox[0]:outBox[2],outBox[1]:outBox[3]])
    return boxes

trLabels1, trImages1, valLabels1, valImages1 = importData('1')

#meanClassify(trImages1, valImages1, trLabels1, valLabels1)
#meanClassify(boxImages(trImages1), boxImages(valImages1), trLabels1, valLabels1)


trLabels2, trImages2, valLabels2, valImages2 = importData('2')

#meanClassify(trImages2, valImages2, trLabels2, valLabels2)
meanClassify(boxImages(trImages2), boxImages(valImages2), trLabels2, valLabels2)






plt.imshow(boxImages(trImages2)[15], cmap='Greys_r')
