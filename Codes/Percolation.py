# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:16:10 2021

@author: Carolina
"""

from Graph import *
import NetGraphics
import matplotlib.pyplot as plt
import numpy as np
rang=np.linspace(0.05,1,20)
rang=np.round(rang, 3)
for i in rang:# this was added so i can save images quickly
 UndirectedGraph.PlotRectReal(i,1,sizex=36,sizey=5,scale=15.74)
#
