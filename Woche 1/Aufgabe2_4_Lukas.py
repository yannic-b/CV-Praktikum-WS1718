# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Yannic, Lukas
"""

###Initialisieren###
Anzahl2Euro = 0
Anzahl1Euro = 0
Anzahl50Cent = 0
Anzahl20Cent = 0
Anzahl10Cent = 0
Anzahl5Cent = 0
Anzahl2Cent= 0
Anzahl1Cent = 0

####Berechnung###

#Eingabe des Geldbetrags
betrag = int(raw_input("Was kostet das Essen: "))

#Anteil der 2-Euro-Stücke
Anzahl2Euro = betrag / 200
betrag %= 200

#Anteil der 1-Euro-Stücke
Anzahl1Euro = betrag / 100
betrag %= 100

#Anteil der 50-Cent-Stücke
Anzahl50Cent = betrag / 50
betrag %= 50

#Anteil der 20-Cent-Stücke
Anzahl20Cent = betrag / 20
betrag %= 20

#Anteil der 10-Cent-Stücke
Anzahl10Cent = betrag / 10
betrag %= 10

#Anteil der 5-Cent-Stücke
Anzahl5Cent = betrag / 5
betrag %= 5

#Anzahl der 2-Cent-Stücke
Anzahl2Cent = betrag / 2
betrag %= 2

#Anzahl der 1-Cent-Stücke
Anzahl1Cent = betrag / 1
betrag %= 1

###Ausgabe der Werte###

print "200: " + str(Anzahl2Euro)
print "100: " + str(Anzahl1Euro)
print "50: " + str(Anzahl50Cent)
print "20: " + str(Anzahl20Cent)
print "10: " + str(Anzahl10Cent)
print "5: " + str(Anzahl5Cent)
print "2: " + str(Anzahl2Cent)
print "1: " + str(Anzahl1Cent)