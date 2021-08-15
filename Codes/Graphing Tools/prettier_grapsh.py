# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 16:22:23 2021

@author: Carolina
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data = pd.read_csv(r"C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp\videos y fotos medidas\Circular\excel\mismo_desplazamiento.csv", na_filter= False)
display(data)
p=np.array(data['p'])
w=np.array(data['m'])*9.8/1000
y=-np.array(data['y'])
p_c=0.5
fig, (ax1) = plt.subplots(1, 1,figsize=(16,8))#aca puedes cambiar el tamaño de la figura 12 es el ancho y 8 la altura
ax1.plot(p, w, linestyle='',marker='o',markersize=9,color='black',mfc='red')
ax1.set_xlabel(r'$probabilidad$',fontsize=30)
ax1.set_ylabel(r'peso$[N]\pm 0.0098$',fontsize=30)
ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
ax1.tick_params( labelsize=20)
plt.show()
plt.close()
logs_p=np.log(np.abs(p-p_c))
logs_w=np.log(w)
fig, (ax1) = plt.subplots(1, 1,figsize=(16,8))#aca puedes cambiar el tamaño de la figura 12 es el ancho y 8 la altura
ax1.plot(logs_p, logs_w, linestyle='',marker='o',markersize=9,color='black',mfc='red')