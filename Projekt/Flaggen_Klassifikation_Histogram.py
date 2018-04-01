'''
Fotos von Flaggen vergleichen im klassischen Ansatz

@author: Lukas

'''



import numpy as np
from skimage import io
import glob
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from skimage.transform import rotate
import pandas as pd
from scipy.misc import imsave
#import cv2
import time




### Daten importieren
timestamp1 = time.time()

trPaths = glob.glob('./data/train/*.png')
trDescriptors =  []
trLabels = []

vaPaths = glob.glob('./data/validation/*.png')
vaDescriptors =  []
vaLabels = []

otsu_correction = 47               #Wert, der bei der Otus-Berechnung hinzuaddiert wrid.
#n=0                               #Laufnummer zum Abspeichern der Bilder

def imageProcessing(mode, img):
    ### RGB ###    
    
    imgG = img[:,:,0] * 0.333 + img[:,:,1] * 0.333 + img[:,:,2] * 0.333
    imgG = imgG.astype(np.int)
    mask = imgG < threshold_otsu(imgG) + otsu_correction
    #imsave("mask_otsu" + str(n) + ".png", mask) #speichert die einzelnen Kacheln mit einer Laufnummer
    #global n
    #n = n + 1
    
    ### HSV ###
    #img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    #blue_lower = np.array([90,33,50])
    #blue_upper = np.array([126,255,255])
    #mask = cv2.inRange(img_hsv,blue_lower,blue_upper)    
    
    #der Drehwinkel, der das Objekt achsenparallel ausrichtet laesst sich ueber die minimale Flaeche der Bounding Box ermitteln
    minArea = np.inf #minimale ist am Anfang unendlich
    bestI = 0 #bester Drehwinkel ist 0
    for i in range(90): #Winkel von 0 bis 89 Grad durgehen, alle groesseren sind nicht relevant
        maskR = rotate(mask, i, order =0) #Maske drehen,dabei ist order = 0 wichtig, damit es weiterhin nur 0 und 1 in der Maske gibt, sonst wuerde interpoliert werden und das Binaerbild zerstoert werden
        props = regionprops(maskR.astype(np.int))[0]
        xMin,yMin,xMax,yMax = props.bbox
        if (xMax-xMin)*(yMax-yMin) < minArea: #Flaeche des Bounding Box berechnen und Vergleichen
            minArea = (xMax-xMin)*(yMax-yMin) #ggf neue minimale Fläche und i merken
            bestI = i
    mask = rotate(mask, bestI, order =0) #Rotation mit bestem i durchführen -> Maskeist jetzt achsenparallel ausgerichtet
    props = regionprops(mask.astype(np.int))[0]
    xMin,yMin,xMax,yMax = props.bbox
    crop = img[xMin:xMax,yMin:yMax,:] #Bild wird auf kleinsten Bereich zugeschnitten
    
    h = crop.shape[0]       #Höhe des zugeschnittenen Bildes
    w = crop.shape[1]       #Breite des zugeschnittenen Bildes

    Kachelhistlist = []     #Array, dass die Histogramme der Kacheln eines Bildes enthält

    for i in range(5): #i = 0-4
        for j in range(5):
            
            cropKachel = crop[int(i*h/5.0):int((i+1)*h/5.0),int(j*w/5.0):int((j+1)*w/5.0)] # Kachelung in 5 x 5 
            cropKachelReshaped = cropKachel.reshape((cropKachel.shape[0]*cropKachel.shape[1],3)) #Reshapen der Kacheln fürs Histogramm
            Kachelhistlist.append(np.histogramdd(cropKachelReshaped, bins = (2,2,2), range = ((0,256),(0,256),(0,256)), normed = True)[0])
            #imsave("Kachel" + str(n) + ".png", cropKachel) #speichert die einzelnen Kacheln mit einer Laufnummer
            #global n
            #n = n + 1

            
    if mode == "train":
        trDescriptors.append(np.array(Kachelhistlist))
        imageComparison("train")
    else:
        vaDescriptors.append(np.array(Kachelhistlist))
        imageComparison("val")
        


def imageComparison(mode):                  #Zuordnung der Bilder zu den Ländern
    if mode == "train":
        labels = trLabels
    else:
        labels = vaLabels
    
    if labelString == 'canada':
        labels.append(0)
    elif labelString == 'china':
        labels.append(1)
    elif labelString == 'france':
        labels.append(2)
    elif labelString == 'germany':
        labels.append(3)
    elif labelString == 'india':
        labels.append(4)
    elif labelString == 'italy':
        labels.append(5)
    elif labelString == 'japan':
        labels.append(6)
    elif labelString == 'russia':
        labels.append(7)
    elif labelString == 'uk':
        labels.append(8)
    elif labelString == 'usa':
        labels.append(9)
    


### für alle Trainingsdaten

for imgPath in trPaths:                                             #für jedes Trainingsbild
    labelString = imgPath.split('\\')[-1].split('.')[0][0:-2]       #Splitte den Dateipfad um das Land herauszufinden
    img = io.imread(imgPath)[:,:,:3]                                #Bilder einlesen
    imageProcessing("train",img)                                    #starte Methode imageProcessing
    

### für alle Validierungsdaten
        
for imgPath in vaPaths:
    labelString = imgPath.split('\\')[-1].split('.')[0][0:-2]
    img = io.imread(imgPath)[:,:,:3]
    imageProcessing("val", img)

### Berechnung mittels Euklidischer Distanz, welches Trainingsbild einem Validierungsbild am nächsten kommt
### k-nearest neighbor    
    
result = []
for vaDesc in vaDescriptors:                                        #für jedes Validierungsbild
    distances = []                                  
    for trDesc in trDescriptors:                                    #für jedes Trainingsbild
        distances.append(np.linalg.norm(vaDesc - trDesc))           #füge die Differenzen der Histogramme der Validierungs- und Trainingsbilder ins Array hinzu
    result.append(trLabels[np.argmin(distances)])                   #füge in results die kürzeste Distanz hinzu

### Berechnung des Anteil der richtigen Zuordnungen
    
correct = 0.0
for r,l in zip(result, vaLabels):                                   #überprüfe ob die Kombination von result und vaLabel 
    if r == l:                                                      #gleich ist, dann
        correct+=1                                                  #erhöhe correct um den Wert 1
print "Richtig zugeordnete Validierungsbilder: " + str(correct/len(vaLabels)) + "%"         #teile die Anzahl korrekter Zuweisungen durch die Anzahl aller Validierungslabels
print
   
### Confusionn Matrix

y_actu = pd.Series(vaLabels, name='Actual')
y_pred = pd.Series(result, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
print df_confusion

timestamp2 = time.time()
print timestamp2-timestamp1