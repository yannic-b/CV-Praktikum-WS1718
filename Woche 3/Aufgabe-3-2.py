#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 17:03:33 2017

@author: Yannic
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

# Aufgab 1
lenna = misc.imread('./Lenna.png')

pixel = 512*512

lennaR = lenna[:,:,0] #-> Rot-Kanal des ersten Bildes
lennaG = lenna[:,:,1] #-> Blau-Kanal des ersten Bildes
lennaB = lenna[:,:,2] #-> Grün-Kanal des ersten Bildes

lennaGrey = (lennaR.astype(np.int64)+lennaG.astype(np.int64)+lennaB.astype(np.int64)) / 3

# 3.1.4
lennaInverted = -lenna

# 3.1.5
lennaMean = np.mean(lenna, axis=(0,1))
lennaStd = np.std(lenna, axis=(0,1))
print lennaMean, lennaStd

# 3.1.6
histR = np.histogram(lennaR, bins = 64, range = (0,256))[0]
histG = np.histogram(lennaG, bins = 64, range = (0,256))[0]
histB = np.histogram(lennaB, bins = 64, range = (0,256))[0]

# lennaHist = np.histogramdd(lenna, bins = [4,4,4], range=((0,256),(0,256),(0,256)))[0]
# print histR, histG, histB
# print lennaHist

# plt.imshow(lennaInverted, cmap='Greys_r')


# Aufgabe 2
d = np.load('./trainingsDatenFarbe.npz')
trImgs = d['data']
trLabels = d['labels']

imageMeans = []
for img in trImgs:
    means = []
    means.append(img[:1024].mean()) #-> Rot-Kanal des ersten Bildes
    means.append(img[1024:2048].mean()) #-> Blau-Kanal des ersten Bildes
    means.append(img[2048:].mean()) #-> Grün-Kanal des ersten Bildes
    imageMeans.append(means)
    
imageStds = []
for img in trImgs:
    stds = []
    stds.append(img[:1024].std()) #-> Rot-Kanal des ersten Bildes
    stds.append(img[1024:2048].std()) #-> Blau-Kanal des ersten Bildes
    stds.append(img[2048:].std()) #-> Grün-Kanal des ersten Bildes
    imageStds.append(stds)
    
imgCombined = np.array(list(zip(imageMeans, imageStds)))

# print imageCombined

dVal = np.load('./validierungsDatenFarbe.npz')
valImgs = dVal['data']
valLabels = dVal['labels']

imageMeansVal = []
for img in valImgs:
    means = []
    means.append(img[:1024].mean()) #-> Rot-Kanal des ersten Bildes
    means.append(img[1024:2048].mean()) #-> Blau-Kanal des ersten Bildes
    means.append(img[2048:].mean()) #-> Grün-Kanal des ersten Bildes
    imageMeansVal.append(means)
    
imageStdsVal = []
for img in valImgs:
    stds = []
    stds.append(img[:1024].std()) #-> Rot-Kanal des ersten Bildes
    stds.append(img[1024:2048].std()) #-> Blau-Kanal des ersten Bildes
    stds.append(img[2048:].std()) #-> Grün-Kanal des ersten Bildes
    imageStdsVal.append(stds)
    
valImgCombined = np.array(list(zip(imageMeansVal, imageStdsVal)))

def l2Distance(img1, img2):
    return np.sqrt(np.square(img1[0] - img2[0]) + np.square(img1[1] - img2[1]))

testLabels = np.zeros(30)
for i in range(valImgCombined.shape[0]):
    minDistance = 256
    for j in range(imgCombined.shape[0]):
        if np.all(l2Distance(valImgCombined[i], imgCombined[j]) < minDistance):
            minDistance = l2Distance(valImgCombined[i], imgCombined[j])
            testLabels[i] = trLabels[j]
            
# print testLabels

# print np.array(list(zip(testLabels, valLabels)))    

# Aufgabe 3

def histL2Distance1D(img1, img2, b=16):
    histR1 = np.histogram(img1[:1024], bins = b, range = (0,256))[0]
    histG1 = np.histogram(img1[1024:2048], bins = b, range = (0,256))[0]
    histB1 = np.histogram(img1[2048:], bins = b, range = (0,256))[0]
    histR2 = np.histogram(img2[:1024], bins = b, range = (0,256))[0]
    histG2 = np.histogram(img2[1024:2048], bins = b, range = (0,256))[0]
    histB2 = np.histogram(img2[2048:], bins = b, range = (0,256))[0]
    sq1 = np.square(histR1-histR2)
    sq2 = np.square(histG1-histG2)
    sq3 = np.square(histB1-histB2)
    out = sq1+sq2+sq3
    return np.sum(out)

def histL2Distance3D(img1, img2):
    hist1 = np.histogramdd(img1, bins = [8,8,8], range=((0,256),(0,256),(0,256)))[0]
    hist2 = np.histogramdd(img2, bins = [8,8,8], range=((0,256),(0,256),(0,256)))[0]
    return (np.sum(np.square(hist1 - hist2)))

testLabels2 = np.zeros(30)
for i in range(valImgCombined.shape[0]):
    minDistance = 256
    for j in range(imgCombined.shape[0]):
        if np.all(histL2Distance1D(valImgCombined[i], imgCombined[j]) < minDistance):
            minDistance = histL2Distance1D(valImgCombined[i], imgCombined[j])
            testLabels2[i] = trLabels[j]
            
correct2 = 0
for x in range(testLabels2.size):
    if testLabels2[x] == valLabels[x]:
        correct2 += 1
print correct2           
# print np.array(list(zip(testLabels2, valLabels)))

testLabels3 = np.zeros(30)
for i in range(valImgCombined.shape[0]):
    minDistance = 256
    for j in range(imgCombined.shape[0]):
        if histL2Distance3D(valImgCombined[i], imgCombined[j]) < minDistance:
            minDistance = histL2Distance3D(valImgCombined[i], imgCombined[j])
            testLabels3[i] = trLabels[j]

correct3 = 0
for x in range(testLabels3.size):
    if testLabels3[x] == valLabels[x]:
        correct3 += 1 
print correct3
            
# print np.array(list(zip(testLabels3, valLabels)))


# Aufgabe 4

testLabels4 = np.zeros(30)
for i in range(valImgCombined.shape[0]):
    distances = []
    for j in range(imgCombined.shape[0]):
        array = [histL2Distance3D(valImgCombined[i], imgCombined[j]), i, j]
        distances.append(array)
    distances.sort(key=lambda x: x[0])
    counter = [[0, 1], [0, 4], [0, 8]]
    for x in range(1):
        if distances[x][2] == 1:
            counter[0][0] += 1
        elif distances[x][2] == 4:
            counter[1][0] += 1
        elif distances[x][2] == 8:
            counter[2][0] += 1
    counter.sort(key=lambda x: x[0])
    testLabels4[i] = counter[0][1]

correct4 = 0
for x in range(testLabels4.size):
    if testLabels4[x] == valLabels[x]:
        correct4 += 1 
print correct4














