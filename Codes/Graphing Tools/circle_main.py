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


cyful= cycle(('none','full'))
marker = cycle(('s', '+', 'd', 'o', '*')) 
fig, (ax1)=plt.subplots(1,1,figsize=(16,8))
path=r'C:\Users\acmor\Desktop\excel_semilla'
probs=[1,2,3,4]
columns=['pendiente','intercepto','R','BSE','last','lastLog','lastLogerr','E','logE','Eerror','logEerror','p_err']
df=showcase.init_tab(probs,columns)
stops=[10,10,10,10]
names=['1','2','3','4']
pc_real=0.91
pc_ideal=0.58
pc_real_dev=0.08
pc_ideal_dev=0.02
#---------------------P=0.0--------------------------------------------------------------------
for i in range(4):
 p0=showcase()
 stop0=stops[i]
 X0,Y0,weight0,height0=p0.returns(os.path.join(path,'medida'+names[i]+'_good.csv'),1,stop0,probs[i])
 ax1.errorbar(weight0,height0,marker=next(marker),ls=' ',ms=8,c='black',fillstyle=next(cyful), markeredgewidth=2)
 ax1.text(weight0[len(weight0)-1]+0.025,height0[len(height0)-1]-0.0025,str(probs[i]),fontsize=21)
 ax1.plot(X0,Y0,color='black',ls='dashed')
 ax1.set_xlim(0,3.6)
 p0.tab(stop0,probs[i],df)
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
showcase.plot_youngs(df,probs)
#showcase.plot_youngs_log_log(df,probs,pc_real,pc_ideal)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None) 
display(df)
df.to_csv(r'C:\Users\acmor\Desktop\excel_semilla\regression.csv')
#-----------------------------------------------------------------
# Young modulus graphics,with log log, find linear regression
#-----------------------------------------------------------------
