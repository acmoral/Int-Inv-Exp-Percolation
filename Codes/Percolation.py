
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:16:10 2021

@author: Carolina
"""
import pandas as pd 
from Graph import *
from aux_theory_functions import *
import NetGraphics
import matplotlib.pyplot as plt
import numpy as np
sizex=50
sizey=4
scale=23.6220
path=r"C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp\videos y fotos medidas\Results"
#-------------------------- crear láminas circulares---------------------------
seed=[0,2,3,20,40,60,80,100,120,140]
probs=[0,0.01,0.02,0.03,0.06,0.09,0.12,0.15,0.18,0.21]
i=0
for p in probs:
    UndirectedGraph.PlotRectReal(p, seed[i], sizex,sizey,scale,bars=True) 
    i+=1