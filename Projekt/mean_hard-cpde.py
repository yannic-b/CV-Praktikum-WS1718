# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 23:54:32 2018

@author: Lukas
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 03 22:22:12 2018

@author: Lukas
"""

import math
from glob import glob
import numpy as np
from scipy import misc, ndimage
from skimage import filters, measure
import matplotlib.pyplot as plt

#glob sucht alle Dateipfade, die zu dem gegebenen Pattern passen,
#wobei der Stern durch beliebig viele beliebige Zeichen ersetzt werden kann
trPaths = glob('.\\flagsTrain\*.png')
trDescriptors =  []
trLabels = []

valPaths = glob('.\\flagsVal\*.png')
valDescriptors =  []
valLabels = []

#Verarbeitung der Trainingsbilder
for imgPath in trPaths:
    #splitten, um Label aus Dateinamen zu bekommen 
    #\\ weil Windows-Pfad + spezielle Belegung von \
    labelString = imgPath.split('\\')[-1].split('.')[0].split('_')[0]
    img = misc.imread(imgPath) # Bild wird geladen

    #trDescriptors.append(np.mean(img,axis=(0,1)))
    trDescriptors.append(np.histogramdd(img.reshape(img.shape[0]*img.shape[1],3), bins = (8,8,8), range=((0,256),(0,256),(0,256)))[0].flatten())

    if labelString == 'canada':
        trLabels.append(0)
    elif labelString == 'china':
        trLabels.append(1)
    elif labelString == 'france':
        trLabels.append(2)
    elif labelString == 'germany':
        trLabels.append(3)
    elif labelString == 'india':
        trLabels.append(4)
    elif labelString == 'italy':
        trLabels.append(5)
    elif labelString == 'japan':
        trLabels.append(6)
    elif labelString == 'russia':
        trLabels.append(7)
    elif labelString == 'uk':
        trLabels.append(8)
    elif labelString == 'usa':
        trLabels.append(9)

#Verarbeitung der Trainingsbilder
for imgPath in valPaths:
    labelString = imgPath.split('\\')[-1].split('.')[0].split('_')[0]
    img = misc.imread(imgPath) # Bild wird geladen

    #valDescriptors.append(np.mean(img,axis=(0,1)))
    valDescriptors.append(np.histogramdd(img.reshape(img.shape[0]*img.shape[1],3), bins = (8,8,8), range=((0,256),(0,256),(0,256)))[0].flatten())
    if labelString == 'canada':
        valLabels.append(0)
    elif labelString == 'china':
        valLabels.append(1)
    elif labelString == 'france':
        valLabels.append(2)
    elif labelString == 'germany':
        valLabels.append(3)
    elif labelString == 'india':
        valLabels.append(4)
    elif labelString == 'italy':
        valLabels.append(5)
    elif labelString == 'japan':
        valLabels.append(6)
    elif labelString == 'russia':
        valLabels.append(7)
    elif labelString == 'uk':
        valLabels.append(8)
    elif labelString == 'usa':
        valLabels.append(9)
        
result = []
for valDesc in valDescriptors:
    distances = []
    for trDesc in trDescriptors:
        distances.append(np.linalg.norm(valDesc-trDesc))
    result.append(trLabels[np.argmin(distances)])

correct = 0.0
for r,l in zip(result, valLabels):
    if r == l:
        correct+=1
        result = correct/len(valLabels)

print correct, "von", len(valLabels), "(", (correct/len(valLabels) * 100), "% )"