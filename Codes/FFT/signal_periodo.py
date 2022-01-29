# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 16:15:35 2022

@author: acmor
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
import pylab as pyl
path=r'C:\Users\acmor\Desktop\vibrations\medidas_2\corner'
class signal:
    def __init__(self,file,stop,p): 
        self.name=file
        self.stop=stop
        self.p=p
        self.y=[]
        self.t=[]
        self.peaks=[]
        self.freq=0
        self.minimos=[]
        self.amp=0
        self.ex=0
    def cut(self):
        self.t=self.t-self.t[0]
        self.y=self.y-self.y[-1]
    def read(self):
        df=pd.read_csv(path+self.name,delimiter=';',engine='python')
        df.drop(df.index[:self.stop], inplace=True)
        self.y=np.array(df['y'])
        self.t=np.array(df['t'])
        self.cut()
        self.peaks, _ = find_peaks(self.y, height=0)
    def frequency(self):
        freq = self.peaks[0:11]
        omega=0
        for i in range(9):
            omega+=self.t[freq[i+1]]-self.t[freq[i]]
        self.freq= len(freq)*2*np.pi/omega
    def amplitud(self):
        self.minimos = argrelextrema(self.y, np.less)
        self.amp=self.y[self.peaks[0]]-self.y[self.minimos[0][0]]
    def exponential_fit(self):
        fit_Y=self.y[self.peaks]
        fit_X=self.t[self.peaks]
        log_Y=np.log(fit_Y)
        params, cv = scipy.optimize.curve_fit(Exp, fit_X, fit_Y)
        m, t, b= params
        self.ex=t
        return params
 
    def pltall(self,numb):
        fig = pyl.figure(figsize=(20,10))
        pyl.subplot(numb)
        pyl.plot(self.t, self.y)
        pyl.plot(self.t[self.peaks[0:10]], self.y[self.peaks[0:10]],'x')
        pyl.xlabel('Tiempo [S]')
        pyl.ylabel('Amplitud [m]')
        pyl.title(str(self.p))
def Exp(x, m, t, b):
    return m * np.exp(-t * x) + b

if __name__ == "__main__":
#-------------------------------------------------------------------------
#              data and global things
#---------------------------------------------------------------------------
   names=[r'\0.csv',r'\0.03.csv',r'\0.06.csv',r'\0.09.csv']#,r'\0.09.csv',r'\0.12.csv',r'\0.15.csv',r'\0.18.csv',r'\0.21.csv']
   stops=[30,30,30,60]
   p=[0,0.03,0.06,0.09]
   n=[421,423,425,427,422,424,426,428]
   sign=[signal(i,stops[names.index(i)],p[names.index(i)]) for i in names]
   x=np.linspace(0, 5,100)
   i=0
   fig = pyl.figure(figsize=(12,15))
#-------------------------------------------------------------------------
#               plotting the waves
#---------------------------------------------------------------------------
   for s in sign:
       s.read()
       s.frequency()
       ax=pyl.subplot(n[i])
       ax.plot(s.t, s.y)
       s.amplitud()
       m, t, b = s.exponential_fit()
       ax.plot(x,Exp(x, m, t, b))
       ax.plot(s.t[s.peaks], s.y[s.peaks],'x')
       ax.plot(s.t[s.minimos[0][0:10]], s.y[s.minimos[0][0:10]],'x')
       ax.set_title(str(s.p))
       ax.set_xlim(0,7)
       i+=1
   fig.text(0.5, 0.04, 'amplitud [m]', ha='center',fontsize=20)
   fig.text(0.04, 0.5, 'tiempo [s]', va='center', rotation='vertical',fontsize=20)
#-------------------------------------------------------------------------
#               plotting the frequency vs p
#--------------------------------------------------------------------------- 
   fig = plt.figure(figsize=(10,10))
   plt.plot(p,[s.freq for s in sign],'.',markersize=20)
   plt.xlabel('P',fontsize=20)
   plt.ylabel('frecuencia',fontsize=20)
#-------------------------------------------------------------------------
#               plotting the first amplitud vs p
#--------------------------------------------------------------------------- 
   fig = plt.figure(figsize=(10,10))
   plt.plot(p,[s.y[0] for s in sign],'.',markersize=20)
   plt.xlabel('P',fontsize=20)
   plt.ylabel('1era Amplitud',fontsize=20)
#-------------------------------------------------------------------------
#               plotting the decay exponent vs p
#--------------------------------------------------------------------------- 
   fig = plt.figure(figsize=(10,10))
   plt.plot(p,[s.ex for s in sign],'.',markersize=20)
   plt.xlabel('P',fontsize=20)
   plt.ylabel('exponente',fontsize=20)