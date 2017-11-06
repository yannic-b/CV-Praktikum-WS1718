#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:09:58 2017

@author: Yannic
"""

out = ""
zeile1 = "\nAuf der Mauer, auf der Lauer"
zeile2 = "\nsitzt ’ne kleine "
zeile5start = "\nSeht euch nur die "
zeile5ende = "an,"
zeile6start = "\nwie die "
zeile6ende = "kann!"

def insWanze(vers):
    return "Wanze"[:(5-vers)]

def insTanzen(vers):
    if vers > 0:
        return "tanzen"[:(5-vers)]
    else:
        return "tanzen"
    
def insDots(vers, innen):
    if vers > 0:
        return "... "
    else:
        if innen:
            return " "
        else:
            return ". "
        
for i in xrange(6):
    for _ in xrange(2):
        out += zeile1 + zeile2 + insWanze(i) + insDots(i, 0)
    out += zeile5start + insWanze(i) + insDots(i, 1) + zeile5ende
    out += zeile6start + insWanze(i) + insDots(i, 1) + insTanzen(i) + insDots(i, 1) + zeile6ende
    out += zeile1 + zeile2 + insWanze(i) + insDots(i, 0) + "\n"
    
print out