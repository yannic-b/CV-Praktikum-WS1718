# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 01:44:25 2018

@author: Lukas
"""

import numpy as np
from skimage import io
import glob
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from skimage.transform import rotate
from scipy.misc import imsave
import matplotlib.pyplot as plt

trPaths = glob.glob('./flagsTrain/*.png')
trDescriptors =  []
trLabels = []

vaPaths = glob.glob('./flagsVal/*.png')
vaDescriptors =  []
vaLabels = []

otsu_correction = 100               #Wert, der bei der Otus-Berechnung hinzuaddiert wrid.

mittelwerteKachelnTrain = []        #Hier werden die Mittelwerte der Kacheln des eingelesende Bild zwischengespeichert
n = 0                               #Laufindex für Crop-Bild-Ausgabe

for imgPath in trPaths:
    labelString = imgPath.split('\\')[-1].split('_')[0]
    img = io.imread(imgPath)
    
    
    imgG = img[:,:,0] * 0.333 + img[:,:,1] * 0.333 + img[:,:,2] * 0.333
    imgG = imgG.astype(np.int)
    mask = imgG < threshold_otsu(imgG) + otsu_correction
    #der Drehwinkel, der das Objekt achsenparallel ausrichtet laesst sich ueber die minimale Flaeche der Bounding Box ermitteln
    minArea = np.inf #minimale ist am Anfang unendlich
    bestI = 0 #bester Drehwinkel ist 0
    for i in range(90): #Winkel von 0 bis 89 Grad durgehen, alle groesseren sind nicht relevant
        maskR = rotate(mask, i, order =0) #Maske drehen,dabei ist order = 0 wichtig, damit es weiterhin nur 0 und 1 in der Maske gibt, sonst wuerde interpoliert werden und das Binaerbild zerstoert werden
        props = regionprops(maskR.astype(np.int))[0]
        xMin,yMin,xMax,yMax = props.bbox
        if (xMax-xMin)*(yMax-yMin) < minArea: #Flaeche des Bounding Box berechnen und Vergleichen
            minArea = (xMax-xMin)*(yMax-yMin) #ggf neueminimale Fläche und i merken
            bestI = i
    mask = rotate(mask, bestI, order =0) #Rotation mit bestem i durchführen -> Maskeist jetzt achsenparallel ausgerichtet
    props = regionprops(mask.astype(np.int))[0]
    xMin,yMin,xMax,yMax = props.bbox
    crop = img[xMin:xMax,yMin:yMax,:] #Bild wird auf kleinsten Bereich zugeschnitten
    
    #imsave("trCrop" + str(n) + ".png", crop) #speichert die zugeschnittenen Bilder mit einer Laufnummer
    n = n + 1
    ####
    
    h = crop.shape[0]
    w = crop.shape[1]
    
    for i in range(3): #i = 0,1,2
        for j in range(3):
            cropKachel = crop[int(i*h/3.0):int((i+1)*h/3.0),int(j*w/3.0):int((j+1)*w/3.0)] # Kachelung in 3 x 3 
            #cropKachelReshaped = cropKachel.reshape((cropKachel.shape[0]*cropKachel.shape[1],3))
            mittelwerteKachelnTrain.append(np.mean(cropKachel)) 
            #trDescriptors.append(np.histogramdd(cropKachelReshaped, bins = (8,8,8), range=((0,256),(0,256),(0,256)))[0])
            
            
    trDescriptors.append(np.array(mittelwerteKachelnTrain))
    #trDescriptors.append(np.histogramdd(crop, bins = (8,8,8), range=((0,256),(0,256),(0,256)))[0].flatten())
    mittelwerteKachelnTrain = []


    if labelString == 'canada':
        trLabels.append(0)
    elif labelString == 'china':
        trLabels.append(1)
    elif labelString == 'france':
        trLabels.append(2)


###
###Validierungsdaten
###
        
mittelwerteKachelnVal = []

        
for imgPath in vaPaths:
    labelString = imgPath.split('\\')[-1].split('_')[0]
    img = io.imread(imgPath)
    
    imgG = img[:,:,0] * 0.333 + img[:,:,1] * 0.333 + img[:,:,2] * 0.333
    imgG = imgG.astype(np.int)
    mask = imgG < threshold_otsu(imgG) + otsu_correction 
    #der Drehwinkel, der das Objekt achsenparallel ausrichtet laesst sich ueber die minimale Flaeche der Bounding Box ermitteln
    minArea = np.inf #minimale ist am Anfang unendlich
    bestI = 0 #bester Drehwinkel ist 0
    for i in range(90): #Winkel von 0 bis 89 Grad durgehen, alle groesseren sind nicht relevant
        maskR = rotate(mask, i, order =0) #Maske drehen,dabei ist order = 0 wichtig, damit es weiterhin nur 0 und 1 in der Maske gibt, sonst wuerde interpoliert werden und das Binaerbild zerstoert werden
        props = regionprops(maskR.astype(np.int))[0]
        xMin,yMin,xMax,yMax = props.bbox
        if (xMax-xMin)*(yMax-yMin) < minArea: #Flaeche des Bounding Box berechnen und Vergleichen
            minArea = (xMax-xMin)*(yMax-yMin) #ggf neueminimale Fläche und i merken
            bestI = i
    mask = rotate(mask, bestI, order =0) #Rotation mit bestem i durchführen -> Maskeist jetzt achsenparallel ausgerichtet
    props = regionprops(mask.astype(np.int))[0]
    xMin,yMin,xMax,yMax = props.bbox
    crop = img[xMin:xMax,yMin:yMax,:]

    
    h = crop.shape[0]
    w = crop.shape[1]
    
    for i in range(3): #i = 0,1,2
        for j in range(3):
            cropKachel = crop[int(i*h/3.0):int((i+1)*h/3.0),int(j*w/3.0):int((j+1)*w/3.0)]
            #cropKachelReshaped = cropKachel.reshape((cropKachel.shape[0]*cropKachel.shape[1],3))
            mittelwerteKachelnVal.append(np.mean(cropKachel))
            #vaDescriptors.append(np.histogramdd(cropKachelReshaped, bins = (8,8,8), range=((0,256),(0,256),(0,256)))[0])
    vaDescriptors.append(np.array(mittelwerteKachelnVal)) #Mean
    
            
    mittelwerteKachelnVal = []
    
    
    if labelString == 'canada':
        vaLabels.append(0)
    elif labelString == 'china':
        vaLabels.append(1)
    elif labelString == 'france':
        vaLabels.append(2)
        

        
result = []
for vaDesc in vaDescriptors:
    distances = []
    for trDesc in trDescriptors:
        distances.append(np.linalg.norm(vaDesc - trDesc))
    result.append(trLabels[np.argmin(distances)])

correct = 0.0
for r,l in zip(result, vaLabels):
    if r == l:
        correct+=1
print correct/len(vaLabels)    