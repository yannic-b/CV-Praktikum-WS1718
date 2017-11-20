# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:50:27 2017

@author: 5kleber
"""

from glob import glob
import numpy as np
from scipy import misc
from skimage import filters, measure
import matplotlib.pyplot as plt

hariTrain = glob('./haribo1/hariboTrain/*.png')

trLabels = []
for string in hariTrain:
    trLabels.append(string.split('/')[-1].split('.')[0].split('_')[0])

trImages = []
for img in hariTrain:
    trImages.append(misc.imread(img))

hariVal = glob('./haribo1/hariboVal/*.png')

valLabels = []
for string in hariVal:
    valLabels.append(string.split('/')[-1].split('.')[0].split('_')[0])

valImages = []
for img in hariVal:
    valImages.append(misc.imread(img))

# def importData()
    
def l2Distance(img1, img2):
    return np.sqrt(np.square(img1[0] - img2[0]) + np.square(img1[1] - img2[1]) + np.square(img1[2] - img2[2]))
    
def calcMean(imgs):
    means = []
    for img in imgs:
        means.append(np.mean(img, axis=(0,1)))
    return np.array(means)

def meanClassify(trImgs, valImgs):
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
        otsu = filters.threshold_otsu(grey)
        binary = (grey <= otsu)
        props = measure.regionprops(binary.astype(int))[0]
        box = props.bbox
        boxes.append(img[box[0]:box[2],box[1]:box[3]])
    return boxes


meanClassify(trImages, valImages)
meanClassify(boxImages(trImages), boxImages(valImages))









plt.imshow(boxImages(trImages)[15], cmap='Greys_r')
