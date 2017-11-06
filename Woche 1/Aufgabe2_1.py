#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 12:19:17 2017

@author: Yannic
"""

z = int(raw_input('Ungerade Zahl: '))

l = range(1,z+2,2)

out = 1

for x in l:
    out *= x
    
print out


