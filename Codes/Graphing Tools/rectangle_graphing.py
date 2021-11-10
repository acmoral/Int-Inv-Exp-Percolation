import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from uncertainties import ufloat
from uncertainties.umath import *
from uncertainties.umath import * 
from itertools import cycle
from matplotlib import ticker
cycol = cycle('bgrcmk')
marker = cycle((',', '+', '.', 'o', '*')) 
I=0.10*0.0025*0.0025*0.0025/12#The second moment of inertia of the material, respect to theZ axis
so=0.10#The initial lenght of the material
soc=so*so*so#cube it

class showcase:
     def __init__(self): 
       self.weights,self.heights,self.Ypred,self.X=[],[],[],[]
       self.pendiente,self.R,self.int,self.BSE,self.last=0,0,0,0,0
     def read(self,path,step):
       df=pd.read_csv(path)
       a=df['y']
       weight=df['m']
       weight=np.array(weight)/1000
       weight=weight*9.8
       height=a
       self.weights=weight
       self.heights=height
       self.last=weight[-1]
     def init_tab(probs,cols):
        return pd.DataFrame(index=probs, columns=cols)
     def returns(self,path,step,stop,prob):
       self.read(path,step)
       self.linear_regression(stop)
       return self.X,self.Ypred,self.weights,self.heights
     def tab(self,stop,prob,data): #tabulates data
         self.linear_regression(stop)
         pendiente=float("{:.6f}".format(float(self.pendiente)))
         data.at[prob,'pendiente']=pendiente
         data.at[prob,'intercepto']=float("{:.6f}".format(float(self.int)))
         data.at[prob,'R']=float("{:.3f}".format(float(self.R)))
         data.at[prob,'BSE']=float("{:.8f}".format(float(self.BSE)))
         data.at[prob,'last']=self.last
         data.at[prob,'lastLog']=log(self.last)
         lasterr=ufloat(self.last,0.00098)
         lasterr=log(lasterr).std_dev
         data.at[prob,'lastLogerr']=lasterr
         E=soc/(3*I*pendiente)
         data.at[prob,'E']=E
         logE=log(E)
         data.at[prob,'logE']=logE
         af=ufloat(0.1,0)#The measures were cut to precision with the machine, some of the material is consumed, but its sistematic in all the plates
         bf=ufloat(0.0025,0)#this measure is given by the manufacturer, given that they are all cut form the same material, no error is given
         If=af*bf*bf*bf/12
         sof=ufloat(0.10,0.001)
         sof=sof*sof*sof
         s=data['pendiente']
         l=data['BSE']
         Ef0=ufloat(s[prob],l[prob])
         Ef0=sof/(3*If*Ef0)
         data.at[prob,'Eerror'] =Ef0.std_dev
         data.loc[prob:,'logEerror']=log(Ef0).std_dev
          
     def linear_regression(self,stop):
         X =np.array(self.weights).reshape(-1,1) 
         X=X[:stop] # values converts it into a numpy array
         Y = np.array(self.heights).reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column basicamente la transpuesta (1,2,3..) 
         Y=Y[:stop]
         linear_regressor = LinearRegression()  # create object for the class
         linear_regressor.fit(X, Y)  # perform linear regression
         Y_pred = linear_regressor.predict(X)  # make predictions
         ols = sm.OLS(Y, X)
         ols_result = ols.fit()
         self.X=X
         self.BSE=ols_result.bse
         self.Ypred=Y_pred
         self.pendiente=linear_regressor.coef_
         self.int=linear_regressor.intercept_
         self.R=linear_regressor.score(X,Y)
     def plot_youngs(data,probs):
       fig, (ax1)=plt.subplots(1, 1,figsize=(16, 8))
       ax1.errorbar(probs,data['E'],xerr=data['p_err'],yerr=data['Eerror'],marker='D',ls='',ms=7,markerfacecolor="None", markeredgewidth=2,color='black')
       ax1.set_xlabel(r'P',fontsize=30)
       ax1.set_ylabel(r'Modulo de Young $[kg/s^2 m]$',fontsize=30) 
       ax1.tick_params(axis='both', labelsize=20)
       ax1.yaxis.offsetText.set_fontsize(20)
       ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
       #for i_x, i_y in zip(probs[:5], data['last'][:5]):
        # ax1.text(i_x, i_y, '({}, {:.2e})'.format(i_x, i_y))
     def plot_youngs_log_log(data,probs,pc_real,pc_ideal):
       fig, (ax1)=plt.subplots(1, 1,figsize=(16, 8))
       p_array=np.array(probs)
       logs_real=np.log(np.abs(p_array-pc_real))
       ax1.errorbar(logs_real,data['logE'],xerr=0,yerr=data['logEerror'],marker='D',ls='',ms=7,markerfacecolor="None", markeredgewidth=2,color='black')
       ax1.set_xlabel(r'$log|p_c-p|$',fontsize=30)
       ax1.set_ylabel(r'$log(E)$',fontsize=30) 
       ax1.tick_params(axis='both', labelsize=20)
       ax1.yaxis.offsetText.set_fontsize(20)
       ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
       X=logs_real.reshape(-1,1)
       Y=np.array(data['logE'])
       Y=Y.reshape(-1,1)
       linear_regressor = LinearRegression()  # create object for the class
       linear_regressor.fit(X, Y)
       Y_pred = linear_regressor.predict(X)
       ax1.plot(X,Y_pred,color='black')
       print(linear_regressor.coef_ )
       print(linear_regressor.intercept_)
       print(linear_regressor.score(X,Y))
     def plot_weights_log_log(data,probs):
       fig, (ax1)=plt.subplots(1, 1,figsize=(10, 10))
       ax=np.array(probs[1:5])
       print(ax)
       logax=np.log(ax)
       print(logax)
       ay=data['lastLog'][1:5]
       ayerr=data['lastLogerr'][1:5]
       by=data['lastLog'][5:]
       byerr=data['lastLogerr'][5:]
       print(len(by),len(ax))
       #ax1.errorbar(x, y, xerr=xerr, yerr=yerr, linestyle=' ',marker='o',markersize=1, c="orange")
       ax1.errorbar(logax,ay,xerr=0.001,yerr=ayerr,marker='o',ls='',ms=5,label='H=1mm')
       #ax1.errorbar(logax,by,xerr=0.001,yerr=byerr,marker='o',ls='',ms=5,color='black',label='H=2mm',markerfacecolor="None", markeredgewidth=2)
       #ax1.plot(probs[0],np.log(data['logE'][0].astype(float)),marker='o',ls='',ms=9,color='black',markerfacecolor="None", markeredgewidth=2)
       ax1.set_xlabel(r'Probability',fontsize=20)
       ax1.set_ylabel(r'Modulo de Young',fontsize=20) 
       ax1.tick_params(axis='both', labelsize=20)
       ax1.yaxis.offsetText.set_fontsize(20)
       ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
       #for i_x, i_y in zip(probs[:5], data['last'][:5]):
        # ax1.text(i_x, i_y, '({}, {:.2e})'.format(i_x, i_y))
       ax1.legend(prop={'size': 20})

