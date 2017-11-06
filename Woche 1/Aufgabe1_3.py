#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:42:18 2017

@author: Yannic
"""

l_dict = {"England":(0.88872,'GBP'),
          "Schweiz":(1.15105,'CHF'),
          "USA":(1.18342,'USD'),
          "Norwegen":(9.31837,'NOK'),
          "Polen":(4.25688,'PLN'),
          "Japan":(132.08176,'JPY')}

land = raw_input("Ein Land bitte: ")
euro = float(raw_input("Wie viel Euro hast du?: "))

print(str(euro * l_dict[land][0]) + " " + l_dict[land][1])


