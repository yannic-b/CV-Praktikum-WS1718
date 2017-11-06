# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:57:03 2017

@author: LKleber
"""
###Initialisierung###
strophe = 1
woerter = ["Wanze", "tanzen"]       #List mit den beiden sich ändernen Wörtern

while (strophe <= 6):               #bis Strophe 6 mache... 
 
    if not (woerter[0] == "" and woerter[1] == ""):         #wenn Wanze und tanzen nicht leer sind
        print "Strophe " + str(strophe) + "\n" + "Auf der Mauer, auf der Lauer\nsitzt ’ne kleine " + woerter[0] + ".\nAuf der Mauer, auf der Lauer\nsitzt ’ne kleine " + woerter[0] + ".\nSeht euch nur die " + woerter[0] + " an, \nwie die " + woerter[0] + " " + woerter[1] + " kann!  \nAuf der Mauer, auf der Lauer  \nsitzt ’ne kleine " + woerter[0] + ". \n"
        
        if woerter[1] == "tanzen":                          #wenn "tanzen" als zweiter Index in List steht,
            woerter[0] = str(woerter[0])[:-1]               #dann lösche den letzen Buchstaben von Element 1
            woerter[1] = str(woerter[1])[:-2]               #und lösche die beiden letzen Buchstaben von Element 2
                                                            #wichtig weil "tanze" im Lied übersprungen wird
        else:
            woerter[0] = str(woerter[0])[:-1]               #ansonsten läsche den letzten Buchstaben von Element 1
            woerter[1] = str(woerter[1])[:-1]               #und den letzten Buchstaben von Element 2

                
    else:                                                   #wenn alle Buchstaben entfernt, ersetze leere Elemente mit ...
        woerter[0] = "..."
        woerter[1] = "..."
        print "Strophe " + str(strophe) + "\n" + "Auf der Mauer, auf der Lauer\nsitzt ’ne kleine " + woerter[0] + ".\nAuf der Mauer, auf der Lauer\nsitzt ’ne kleine " + woerter[0] + ".\nSeht euch nur die " + woerter[0] + " an, \nwie die " + woerter[0] + " " + woerter[1] + " kann!  \nAuf der Mauer, auf der Lauer  \nsitzt ’ne kleine " + woerter[0] + ". \n"
    
    strophe += 1                                            #Strophe hochzählen

