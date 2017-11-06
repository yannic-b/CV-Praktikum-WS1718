#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:08:49 2017

@author: Yannic
"""

### Aufgabe 1:

l5 = [0]*5
l6 = [1]*6
l4 = [2]*4

l = []
l.extend(l5)
l.extend(l6)
l.extend(l4)

s = set(l)
l = list(s)

l10 = range(1,11)

s10 = set(l10)

sunion = s.union(s10)

lunion = list(sunion)
