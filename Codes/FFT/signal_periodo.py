# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 16:15:35 2022

@author: acmor
"""
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import ticker
import scipy.optimize
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
import pylab as pyl
path=r'C:\Users\acmor\Desktop\vibrations\medidas_3'
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
   names=[r'\0.csv',r'\0.01.csv',r'\0.02.csv',r'\0.03.csv',r'\0.06.csv',r'\0.09.csv',r'\0.12.csv',r'\0.15.csv',r'\0.18.csv',r'\0.21.csv']
   stops=[0,0,0,0,200,90,820,665,625,580]
   p=[0,0.01,0.02,0.03,0.06,0.09,0.12,0.15,0.18,0.21]#]
   sign=[signal(i,stops[names.index(i)],p[names.index(i)]) for i in names]
   x=np.linspace(0, 5,100)
   i=1
   fig = pyl.figure(figsize=(15,15))
#-------------------------------------------------------------------------
#               plotting the waves
#---------------------------------------------------------------------------
   for s in sign:
       s.read()
       s.frequency()
       ax=pyl.subplot(5,2,i)
       ax.plot(s.t, s.y,color ='black')
       s.amplitud()
       m, t, b = s.exponential_fit()
       #ax.plot(x,Exp(x, m, t, b))
       ax.plot(s.t[s.peaks], s.y[s.peaks],'x')
       #ax.plot(s.t[s.minimos[0][0:10]], s.y[s.minimos[0][0:10]],'x')
       ax.set_title(str(s.p))
       ax.set_xlim(0,7)
       i+=1
   fig.text(0.5, 0.04,'tiempo [s]' , ha='center',fontsize=20)
   fig.text(0.04, 0.5, 'Amplitud [m]', va='center', rotation='vertical',fontsize=20)
#-------------------------------------------------------------------------
#               plotting the frequency vs p
#--------------------------------------------------------------------------- 
   fig, ax1 =plt.subplots(1, 1,figsize=(16, 10))
   y=[s.freq for s in sign]
   ax1.plot(p,y,'s',markersize=8,color='black')
   print([s.freq for s in sign])
   ax1.set_xlabel(r'Probabilidad',fontsize=20)
   ax1.set_ylabel(r'Frecuencia $[s^{-1}]$',fontsize=20)
   ax1.tick_params(axis='both', labelsize=20)
   ax1.yaxis.offsetText.set_fontsize(20)
   ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
   ax1.minorticks_on()
   ax1.tick_params(which='both', width=2)
   ax1.tick_params(which='minor', length=5)
   ax1.tick_params(which='major', length=8)
   X =np.array(p).reshape(-1,1) 
   Y = np.array(y).reshape(-1,1)
   X=X[2:]
   Y=Y[2:]# -1 means that calculate the dimension of rows, but have 1 column basicamente la transpuesta (1,2,3..) 
   X=np.log(abs(X-0.85))
   Y=np.log(Y)
   linear_regressor = LinearRegression()  # create object for the class
   linear_regressor.fit(X, Y)  # perform linear regression
   Y_pred = linear_regressor.predict(X)  # make predictions
   ols = sm.OLS(Y, X)
   ols_result = ols.fit()
   BSE=ols_result.bse
   Pendiente=linear_regressor.coef_
   Intercepto=linear_regressor.intercept_
   Rcuadrado=linear_regressor.score(X,Y)
   fig, (ax1)=plt.subplots(1, 1,figsize=(16, 10))
   ax1.plot(X,Y,'s',markersize=8,color='black')
   ax1.plot(X,Y_pred,markersize=10,color='black')
   ax1.set_xlabel(r'$Log \enspace |p-p_c|$',fontsize=20)
   ax1.set_ylabel(r'$Log \enspace |f|$',fontsize=20)
   ax1.tick_params(axis='both', labelsize=20)
   ax1.yaxis.offsetText.set_fontsize(20)
   ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
   ax1.minorticks_on()
   ax1.tick_params(which='both', width=2)
   ax1.tick_params(which='minor', length=5)
   ax1.tick_params(which='major', length=8)
   print(Pendiente, Intercepto, Rcuadrado)
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