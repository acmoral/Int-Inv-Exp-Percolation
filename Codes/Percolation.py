
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
sizex=20
sizey=10
scale=23.6220
seed=1
path=r"C:/Users/acmor/Desktop"
#-------------------------- crear láminas circulares---------------------------
probs=[0,0.05,0.1,0.15,0.2,0.25,0.30,0.35,0.4,0.39,0.45]
for p in probs:
    UndirectedGraph.PlotRectReal(p, seed, sizex,sizey,scale,bars=True)
    