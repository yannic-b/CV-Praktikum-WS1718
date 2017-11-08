# -*- coding: utf-8 -*-
import numpy as np

"""
Spyder Editor

This is a temporary script file.
"""
# 3.1.1.1-3
d = np.load('./trainingsDaten.npz')
trImgs = d['data']
trLabels = d['labels']

imgM = []
for img in trImgs:
    imgM.append(img.mean())
imgMeans = np.array(imgM)

imgMeans2 = trImgs.mean(1)
    
imgS = []
for img in trImgs:
    imgS.append(np.std(img))
imgStds = np.array(imgS)
    
imgCombined = np.array(list(zip(imgMeans, imgStds)))



d = np.load('./validierungsDaten.npz')
valImgs = d['data']
valLabels = d['labels']

valImgM = []
for img in valImgs:
    valImgM.append(img.mean())
valImgMeans = np.array(valImgM)
    
valImgMeans2 = trImgs.mean(1)    
    
valImgS = []
for img in valImgs:
    valImgS.append(np.std(img))
valImgStds = np.array(valImgS)
    
valImgCombined = np.array(list(zip(valImgMeans2, valImgStds)))

def l2Distance(img1, img2):
    return np.sqrt(np.square(img1[0] - img2[0]) + np.square(img1[1] - img2[1]))


# 3.1.1.4
testLabels = np.zeros(30)
for i in range(valImgCombined.shape[0]):
    minDistance = 256
    for j in range(imgCombined.shape[0]):
        if l2Distance(valImgCombined[i], imgCombined[j]) < minDistance:
            minDistance = l2Distance(valImgCombined[i], imgCombined[j])
            testLabels[i] = trLabels[j]
              
#print testLabels

# 3.1.1.5
correct = 0
for x in range(testLabels.size):
    if testLabels[x] == valLabels[x]:
        correct += 1
        
print correct      


# 3.1.2
def histL2Distance(img1, img2, bs=64):
    hist1 = np.histogram(img1, bins=bs, range=(0,256))
    hist2 = np.histogram(img2, bins=bs, range=(0,256))
    return (np.sum(np.square(hist1[0] - hist2[0])))
    
testLabels2 = np.zeros(30)
for i in range(valImgCombined.shape[0]):
    minDistance = 256
    for j in range(imgCombined.shape[0]):
        if histL2Distance(valImgCombined[i], imgCombined[j]) < minDistance:
            minDistance = histL2Distance(valImgCombined[i], imgCombined[j])
            testLabels2[i] = trLabels[j]
            
#print testLabels2
#print valLabels
print np.array(list(zip(testLabels2, valLabels)))            
            
correct2 = 0
for x in range(testLabels2.size):
    if testLabels2[x] == valLabels[x]:
        correct2 += 1
        
print correct2 


# 3.1.3
coMat = np.zeros((3, 3))
labels = np.array([1, 4, 8])
for i in range(3):
    for j in range(3):
        for x in range(30):
            if (testLabels[x] == labels[i]) & (valLabels[x] == labels[j]):
                coMat[j, i] += 1
                
print coMat
