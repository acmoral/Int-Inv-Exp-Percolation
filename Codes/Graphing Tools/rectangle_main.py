# -*- coding: utf-8 -*-
"""
Created on Tue May 11 18:34:10 2021

@author: Carolina
"""

from rectangle_graphing import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import os
from itertools import cycle
import pandas as pd
marker = cycle(('s', '+', 'd', 'o', '*')) 
cyful= cycle(('none','full'))
fig, (ax1)=plt.subplots(1,1,figsize=(16,8))
path=r'C:\Users\acmor\Desktop\excel_square'
probs=[0,0.05,0.1,0.15,0.2,0.24,0.28,0.32]
columns=['pendiente','intercepto','R','BSE','last','lastLog','lastLogerr','E','logE','Eerror','logEerror','p_err']
df=showcase.init_tab(probs,columns)
stops=[25,14,20,23,60,40,40,51]
names=['0','0,05','0,1','0,15','0,2','0,24','0,28','0,32']
p_err=[0,0,0,0,0,0,0,0]
df['p_err']=p_err
pc_real=0.79
pc_ideal=0.42
pc_real_dev=0.12
pc_ideal_dev=0.02
#---------------------P=0.0--------------------------------------------------------------------
for i in range(8):
 p0=showcase()
 stop0=stops[i]
 X0,Y0,weight0,height0=p0.returns(os.path.join(path,'p='+names[i]+'.csv'),1,stop0,probs[i])
 ax1.errorbar(weight0,height0,marker=next(marker),ls=' ',ms=8,c='black',fillstyle=next(cyful), markeredgewidth=2)
 ax1.text(weight0[len(weight0)-1]-0.65,height0[len(height0)-1]+0.003,str(probs[i]),fontsize=20)
 ax1.plot(X0,Y0,color='black')
 ax1.set_ylim(0,0.1)
 p0.tab(stop0,probs[i],df)
ax1.minorticks_on()
ax1.tick_params(which='both', width=2)
ax1.tick_params(which='minor', length=5)
ax1.tick_params(which='major', length=8)
#
#---------------------------legends,titles-------------------
#-----------------------------------------------------------
ax1.set_xlabel(r'Peso $[N]$',fontsize=30)
ax1.set_ylabel(r' Flexión en Y $[m] $',fontsize=30)
ax1.tick_params(axis='both', labelsize=20)
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
df.to_csv(r'C:\Users\acmor\Desktop\excel_square\regresion.csv')
#-----------------------------------------------------------------
# Young modulus graphics,with log log, find linear regression
#-----------------------------------------------------------------
