# -*- coding: utf-8 -*-
"""
Created on Tue May 11 18:34:10 2021

@author: Carolina
"""

from rectangle_graphing import *
import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import cycle
import pandas as pd
I=0.1*0.0025*0.0025*0.0025/12#The second moment of inertia of the material, respect to theZ axis
so=0.10#The initial lenght of the material
soc=so*so*so#cube it
cycol = cycle(('darkgoldenrod','navy','purple','grey','teal','black','darkred'))
marker = cycle(('s', '+', 'd', 'o', '*')) 
fig, (ax1)=plt.subplots(1,1,figsize=(16,8))
path=r'C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp\square_square_measures\excel'
probs=[0,0.05,0.1,0.15,0.2,0.24,0.28,0.32]
columns=['pendiente','intercepto','R','BSE','last','lastLog','lastLogerr','E','logE','Eerror','logEerror','p_err']
df=showcase.init_tab(probs,columns)
stops=[25,14,20,23,60,40,40,51]
names=['0','0,05','0,1','0,15','0,2','0,24','0,28','0,32']
p_err=[0,0,0,0,0,0,0,0]
df['p_err']=p_err
pc_real=0.79
pc_ideal=0.42
pc_real_dev=0.03 
pc_ideal_dev=0.15
#---------------------P=0.0--------------------------------------------------------------------
for i in range(8):
 p0=showcase()
 stop0=stops[i]
 X0,Y0,weight0,height0=p0.returns(os.path.join(path,'p='+names[i]+'.csv'),1,stop0,probs[i])
 ax1.errorbar(weight0,height0,marker=next(marker),ls=' ',ms=6,label='P='+str(probs[i]),c=next(cycol),markerfacecolor="None", markeredgewidth=2)
 ax1.plot(X0,Y0,color='black',ls='dashed')
 p0.tab(stop0,probs[i],df)
#
#---------------------------legends,titles-------------------
#-----------------------------------------------------------
ax1.set_xlabel(r'peso $[N] \pm 9.8\times10^{-4}$',fontsize=30)
ax1.set_ylabel(r'Desplazamiento en Y $[m] $',fontsize=30)
ax1.tick_params(axis='both', labelsize=20)
ax1.set_title(r'Ancho= 2.5mm, Hueco= 1.5mm',size=30)
ax1.legend(prop={'size': 20})
ax1.xaxis.offsetText.set_fontsize(20)
#-----------------------------------------------------------------
# Young modulus graphics
#-----------------------------------------------------------------

#append it to the calculated logE
showcase.plot_youngs(df,[0,0.05418,0.10285,0.15404,0.20271,0.24770,0.29316,0.34435])
showcase.plot_youngs_log_log(df,[0,0.05418,0.10285,0.15404,0.20271,0.24770,0.29316,0.34435],pc_real,pc_ideal)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None) 
display(df)
df.to_csv(r'C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp\square_square_measures\excel\regresion.csv')
#-----------------------------------------------------------------
# Young modulus graphics,with log log, find linear regression
#-----------------------------------------------------------------
