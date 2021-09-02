
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:16:10 2021

@author: Carolina
"""
import pandas as pd 
from Graph import *
import aux_theory_functions
import NetGraphics
import matplotlib.pyplot as plt
import numpy as np
sizex=40
sizey=5
scale=15.74
p=0.15
path=r"C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp"
#-------------------------- crear láminas circulares---------------------------
seeds=np.linspace(1,4,4,dtype=int)
probs=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,1]
for seed in seeds:
    UndirectedGraph.PlotRectReal(p, seed, sizex,sizey,scale,bars=False)