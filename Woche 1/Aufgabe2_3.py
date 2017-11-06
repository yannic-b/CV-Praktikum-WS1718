#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 12:31:44 2017

@author: Yannic
"""

#Input:
l = []
i = 1
while (i <= 5):
    x = raw_input("Zahl "+str(i)+": ")
    l.append(x)
    i += 1
    
for x in l:
    l[l.index(x)] = float(x)


#Liste
print("\nListe: " + str(l) + "\n")


#Min,Max,Med
ls = l[:]
ls.sort()
#min
print("Min: " + str(ls[0]) + ", " + str(l.index(ls[0])) + "\n")
#max
print("Miax: " + str(ls[4]) + ", " + str(l.index(ls[4])) + "\n")
#median
print("Median: " + str(ls[2]) + "\n")


#Ungerade/Gerade
ungerade = 0

for x in l:
    if x % 2 != 0:
        ungerade += 1
#ungerade        
print("Ungerade: " + str(ungerade) + "\n")
#gerade
print("Gerade: " + str(5 - ungerade) + "\n")


#Unterschiedlich
print("Unterschiedlich: " + str(len(set(l))) + "\n")

ganz = 0
for x in l:
    if x % 1 != 0:
        ganz += 1
#Ganze Zahlen        
print("Ganze Zahlen: " + str(ganz) + "\n")
#Weitere
print("Weitere: " + str(5 - ganz) + "\n")
