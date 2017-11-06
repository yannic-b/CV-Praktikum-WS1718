#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:28:02 2017

@author: Yannic
"""

cents = int(raw_input("Was kostet das Essen (in Cent)?: "))

coins = [200,100,50,20,10,5,2,1]

for c in coins:
    if cents / c > 0:
        print(str(c) + ": " + str(cents/c))
        cents = cents % c
    else:
        print(str(c) + ": 0")   