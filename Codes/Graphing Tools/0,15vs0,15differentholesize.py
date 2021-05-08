# -*- coding: utf-8 -*-
"""
Created on Sun May  2 17:10:40 2021

@author: Carolina
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
#-------------Import end, begin script-----------------------------------------
#---------------------p=0,A=3mm----------------------------------------
df = pd.read_csv(r'C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp\videos y fotos medidas\excel\p=0,15A=3mmH=1mm.csv')
y=df['y']
height=-y[::4]*100
si=len(height)
weight=np.linspace(0,si,si)*15.64+76.7-15.64
weight[0]=0
weight=weight*980 #gravity
fig, (ax1)=plt.subplots(1,1,figsize=(10, 10))
ax1.errorbar(weight,height,marker='s',ls=' ',ms=6,label=r'Ocupación=2279,Area=2279$mm^2$',color='black')
#ax2.errorbar(weight,height,marker='s',ls=' ',ms=6,label='P=0.0',color='black')
#--------------------p=0,A=4mm,H=2mm LR--------------------------------------
X =np.array(weight).reshape(-1,1) 
X=X[:8] # values converts it into a numpy array
Y = np.array(height).reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column basicamente la transpuesta (1,2,3..) 
Y=Y[:8]
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
ax1.plot(X, Y_pred,color='black',ls='dashed')#esta linea roja si es una regresion
pendiente=linear_regressor.coef_
inter=linear_regressor.intercept_
R=linear_regressor.score(X,Y)
#---------------------p=0,05A=3mmH=1mm----------------------------------------
df = pd.read_csv(r'C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp\videos y fotos medidas\excel\p=0,15H=2mmA=3mm.csv')
y=df['y']
height=-y[::4]*100
height=height[:39]
si=len(height)
weight=np.linspace(0,si,si)*15.64+76.7-15.64
weight[0]=0
weight=weight*980 #gravity
weight=weight[:39]
ax1.errorbar(weight,height,marker='.',ls=' ',ms=9,label= r'Ocupación=571,Area=2284$mm^2$',color='seagreen')
#ax2.errorbar(weight,height,marker='.',ls=' ',ms=9,label= 'P=0.05',color='seagreen')
#--------------------p=0,05A=4mm,H=2mm LR--------------------------------------
X =np.array(weight).reshape(-1,1) 
X=X[:8] # values converts it into a numpy array
Y = np.array(height).reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column basicamente la transpuesta (1,2,3..) 
Y=Y[:8]
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
ax1.plot(X, Y_pred,color='black',ls='dashed')#esta linea roja si es una regresion
pendiente=linear_regressor.coef_
inter=linear_regressor.intercept_
R=linear_regressor.score(X,Y)
print(pendiente, inter)
#---------------------------------------------------------------
ax1.set_xlabel(r'peso $[dyn] \pm 0.006$',fontsize=20)
ax1.set_ylabel(r'Desplazamiento en Y $[cm] $',fontsize=20)
#ax2.set_xlabel(r'peso $[dyn] \pm 0.006$',fontsize=20)
#ax2.set_ylabel(r'Desplazamiento en Y $[cm] $',fontsize=20)
ax1.tick_params(axis='both', labelsize=20)
#ax2.tick_params(axis='both', labelsize=20)
ax1.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
#ax2.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
ax1.legend()