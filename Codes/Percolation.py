
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
pc=0.88
path=r"C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp\videos y fotos medidas\Results"
#-------------------------- crear láminas circulares---------------------------
seeds=np.linspace(1,2000,2000,dtype=int)
pc=[]
pcreal=[]
probs=np.linspace(0.3,0.5,40)
for seed in seeds:
  for p in probs:
    var,preal=pc_calculate(seed,p,sizex,sizey,scale)
    if var:
        pcreal.append(preal)
        pc.append(p)
        break

print(np.mean(np.array(pcreal)))
print(np.mean(np.array(pc)))
d={'seed':seeds,'pc_real':pcreal,'pc_ideal':pc}
df=pd.DataFrame.from_dict(d)
df.to_csv(path+r'\pc_seeds_10x10.csv',index=0)
