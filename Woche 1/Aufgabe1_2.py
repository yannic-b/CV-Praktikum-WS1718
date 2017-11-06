#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:25:05 2017

@author: Yannic
"""

###Aufgabe 2:

l = range(1,10)

input = raw_input("Bitte gib einen Index zwischen 1 und 9 ein: ")

i = int(input)

l1 = l[0:i]
l2 = l[(i+1):10]

l1.reverse()
l2 = l2[::-1]

l[0:i] = l1
l[(i+1):10] = l2

print(l)