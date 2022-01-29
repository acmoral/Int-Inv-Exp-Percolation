# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 16:15:35 2022

@author: acmor
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.fftpack as syfp
import pylab as pyl
path=r'C:\Users\acmor\Desktop\vibrations\medidas\2cm'
class signal:
    def __init__(self,file,stop): 
        self.name=file
        self.stop=stop
        self.sig=None
        self.len=0
        self.FFT=None
        self.freq=[]
    def read(self):
        self.sig=pd.read_csv(path+self.name,delimiter=';',engine='python')
        self.sig.drop(self.sig.index[:self.stop], inplace=True)
        self.len=len(self.sig['t'])
        self.FFT= np.fft.fft(self.sig['y'])
        self.freq= syfp.fftfreq(self.len,0.004166667)
    def pltall(self):
        fig = pyl.figure(figsize=(20,10))
        pyl.subplot(221)
        pyl.plot(self.sig['t'], self.sig['y'])
        pyl.xlabel('Time')
        pyl.ylabel('Amplitude')
        pyl.subplot(222)
        pyl.plot(self.freq[:self.len // 2], np.lib.scimath.log10(np.abs(self.FFT)[:self.len // 2]), '.',ls='dotted')
        pyl.subplot(212)
        pyl.bar(self.freq[:self.len // 2], np.abs(self.FFT)[:self.len // 2], width=1)
        pyl.show()

if __name__ == "__main__":
   names=[r'\2cm_0.csv',r'\2cm_003.csv',r'\2cm_006.csv',r'\2cm_009.csv']
   stops=[675,1050,545,420]
   sign=[]
   j=0
   for i in names:
       sign.append(signal(i,stops[j]))
       j+=1
   for s in sign:
       s.read()
   p0=sign[2]
   p0.pltall()