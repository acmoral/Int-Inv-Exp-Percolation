
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:16:10 2021

@author: Carolina
"""
import pandas as pd 
from Graph import *
import NetGraphics
import matplotlib.pyplot as plt
import numpy as np
sizex=36
sizey=5
scale=15.74
path=r"C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp"
#probs=np.round(probs, 3)
#------------------------------------------------------------------------------
#                             Pc calculate
#------------------------------------------------------------------------------
"""
#Calculates Pc for a given configuration sizex size y
pc=[]
pcreal=[]
seeds=np.linspace(1,20,20,dtype=int)
probs=np.linspace(0,1,20)
for seed in seeds:
  for p in probs:
    var,preal=UndirectedGraph.pc_calculate(seed,p,sizex,sizey,scale)
    if var:
        pcreal.append(preal)
        pc.append(p)
        break

print(np.mean(pc),np.mean(pcreal),np.std(pc),np.std(pcreal))
"""
#------------------------------------------------------------------------------
#                             P_real graph
#------------------------------------------------------------------------------
"""
UndirectedGraph.preal_plt(sizex,sizey,scale)
"""
#------------------------------------------------------------------------------
#                            <S> vs P
#------------------------------------------------------------------------------
#UndirectedGraph.sVSp(sizex,sizey,scale)
#------------------------------------------------------------------------------
#                            connectedness
#------------------------------------------------------------------------------
pc=0.39
UndirectedGraph.connect(sizex,sizey,scale,pc)
#------------------------------------------------------------------------------
                           #ns_vs_s
#------------------------------------------------------------------------------
#UndirectedGraph.ns_vs_s(sizex,sizey,scale)
#------------------------------------------------------------------------------
                           #Sum sns vs p
#------------------------------------------------------------------------------
#UndirectedGraph.snsVSp(sizex,sizey,scale)