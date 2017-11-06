#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 12:12:29 2017

@author: Yannic
"""

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

# 2.2.1: SatBilder
agri3 = misc.imread('./satBilder/agri3.png')
agri6 = misc.imread('./satBilder/agri6.png')
urban1 = misc.imread('./satBilder/urban1.png')

satBilder = (urban1,agri3,agri6)

def meanDifference(img1,img2):
    return np.abs(img1.mean() - img2.mean())

print meanDifference(agri3,agri6)
print meanDifference(agri6,urban1)
print meanDifference(agri3,urban1)

# 2.2.2: SatBilder Distances
def histL2Distance(img1,img2):
    hist1 = np.histogram(img1, bins=8, range=(0,256))
    hist2 = np.histogram(img2, bins=8, range=(0,256))
    return np.sum(np.square(hist1[0] - hist2[0]))

def histIntDistance(img1,img2):
    hist1 = np.histogram(img1, bins=8, range=(0,256))
    hist2 = np.histogram(img2, bins=8, range=(0,256))
    min = np.minimum(hist1[0],hist2[0])
    #plt.plot(hist1[1][:-1],min)
    return (np.sum(min) / float(img1.shape[0]*img1.shape[1]))

# '''
# 2.2.3: SatBilder Distances Comparison
for img1 in satBilder:
    for img2 in satBilder:
        print '\nMean Difference: ' + str(meanDifference(img1,img2))
        print 'Euclidean Distance: ' + str(histL2Distance(img1,img2))
        print 'Intersection: ' + str(histIntDistance(img1,img2))
# '''   
        
# 2.2.4: Flaggen
jugoslawien = misc.imread('./flaggen/jugoslawien.png')
russland = misc.imread('./flaggen/russland.png')

print "Mean Difference - Russland/Jugoslawien: " + str(meanDifference(jugoslawien, russland))

def partitionImageMeans(img, xSplits, ySplits):
    parts = []
    for yPart in np.array_split(img, ySplits, axis=0):
        for part in np.array_split(yPart, xSplits, axis=1):
            parts.append(part.mean())
    return np.array(parts)

print partitionImageMeans(russland, 3, 3)
print partitionImageMeans(jugoslawien, 3, 3)
    
