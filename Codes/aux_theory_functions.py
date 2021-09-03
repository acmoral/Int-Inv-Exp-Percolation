import numpy as np
import matplotlib.ticker as mtick
import pandas as pd 
from sklearn.linear_model import LinearRegression
import scipy 
from Graph import *
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
path=r'C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp'
#----------------Gyration Radious-------------------------------------------
def rmean(g,cl):
     S=len(cl)
     rmean=np.sum(cl,axis=0)/S
     return rmean
#-------------------------------------------------------------------------------
def Gradious(g,cl):
     mean=rmean(g,cl)
     R_i=0
     for i in cl:
       value=i-mean
       R_i+=np.dot(value,value)
     return R_i
#------------------------------------------------------------------------------
#               verify if percolates, either horizontally or vertically
#------------------------------------------------------------------------------
def Notpercolates(g,cl):
     L=g.L
     H=g.H
     if type(cl)==None:
         return True
     for node in cl:
      if node[0]==0:
          for node2 in cl:
              if node2[0]==L-1:
                  return False
      elif node[1]==0:
          for node2 in cl:
              if node2[1]==H-1:
                  return False
     return True
 
#------------------------------------------------------------------------------
#               Calcular pc para diferentes semillas
#------------------------------------------------------------------------------
def pc_calculate(seed,p, sizex,sizey,scale):
     scipy.random.seed(seed)
     g=UndirectedGraph.MakeRectangularSitePercolation(sizex,sizey,scale,p,seed)
     g.Fill()
     clusters=g.FindAllClusters()
     preal=len(g.GetNodes())/(g.L*g.H)
     for cl in clusters:
         if len(cl)>=min(g.L,g.H)*3:
          if not Notpercolates(g,cl):
              return True,preal
     return False,preal
              
 #------------------------------------------------------------------------------
#               Divide the clusters according to size
#------------------------------------------------------------------------------         
def s_sizes(g,clusters):
      d={} # an empty dict that is going to contain keys:sizes and values: the index in clusters list
      i=0
      for cl in clusters:
         lenght=len(cl)
         if Notpercolates(g,cl):
          if lenght in d.keys():
             d[lenght].append(i)
          else:
             d[lenght]=[i]
         i+=1
      if d:
       return d
      else:
       d[0]=[0]
       return d
#-------------------------------------------------------------------------------
def ns(g,clusters):
      d=s_sizes(g,clusters)#dictionary of sizes
      s={}#another dict for the cluster size distribution
      L=g.L
      H=g.H
      for key in d.keys():
       s[key]=len(d[key])/(L*H)
      return s
#-------------------------------------------------------------------------------
def Rs_average(g,clusters):
     s=s_sizes(g,clusters)
     Rs={}
     for key in s.keys():
         if key!=0:
          numbers=s[key]
          ss=0
          for i in numbers:
            ss+=Gradious(g,clusters[i])/key
          Rs[key]=ss/len(numbers)
         else:
             Rs[key]=0
     return Rs
#-------------------------------------------------------------------------------
def eps(g,clusters):
     s=s_sizes(g,clusters)
     n_s=ns(g,clusters)
     R_s=Rs_average(g,clusters)
     uppersum=0
     lowersum=0
     for key in s.keys():
         if key!=0:
          val=key*key*n_s[key]
          uppersum+=val*R_s[key]
          lowersum+=val
     if lowersum!=0:
          epsilon=np.sqrt(uppersum/lowersum)
     else:
          epsilon=0
     return epsilon
#------------------------------------------------------------------------------
#               Is my code doing what i want? 
#------------------------------------------------------------------------------
def verify(seed,p,sizex,sizey,scale):
     scipy.random.seed(seed)
     g=UndirectedGraph.MakeRectangularSitePercolation(sizex,sizey,scale,p,seed)
     L=g.L
     H=g.H
     g.Fill()
     clusters=g.FindAllClusters()
     NetGraphics.DrawSquareNetworkSites(L,H,p,nodelists=clusters,imsizex=sizex,imsizey=sizey,scale=scale,change=False)
     total=len(g.GetNodes())
     N=L*H
     NS=ns(g,clusters)
     S=s_sizes(g,clusters)
     sume=0
     for s in S.keys():
         sume+=s*NS[s]
     if sume*N==total:
         print(sume*N,total)
         return True
     else:
         print(sume*N,total)
         return False
        
#-------------------------------------------------------------------------------
def connect(sizex,sizey,scale,pc):
     seeds=np.linspace(1,20,20,dtype=int)
     eps_0=[]
     dev=[]
     p_r=[]
     p_r_dev=[]
     probs=np.linspace(0,1,50)
     for p in probs:
       Y=[]
       p_real=[]
       for seed in seeds:
         scipy.random.seed(seed)
         g=UndirectedGraph.MakeRectangularSitePercolation(sizex,sizey,scale,p,seed)
         g.Fill()
         clusters=g.FindAllClusters()
         epsilon=eps(g,clusters)
         Y.append(epsilon)
         p_real.append(len(g.GetNodes())/(g.L*g.H))
       eps_0.append(np.mean(Y))
       dev.append(np.std(Y))
       p_r.append(np.mean(p_real))
       p_r_dev.append(np.std(p_real))
     fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(16,8))
     ax1.plot(probs,eps_0,marker='D',linestyle=' ',markersize=6,markerfacecolor="None")
     ax2.errorbar(p_r,eps_0,xerr=p_r_dev,yerr=dev,marker='D',linestyle=' ',markersize=6,markerfacecolor="None")
     ax1.yaxis.set_major_formatter(mtick.FuncFormatter(ticks))
     ax1.xaxis.set_major_formatter(mtick.FuncFormatter(ticks)) 
     ax2.yaxis.set_major_formatter(mtick.FuncFormatter(ticks))
     ax2.xaxis.set_major_formatter(mtick.FuncFormatter(ticks))
     ax_params(ax1,xlabel='p',ylabel=r'$\xi(p)$')
     ax_params(ax2,xlabel=r'$p_{real}$',ylabel=None)
     fig.savefig(path+r'\videos y fotos medidas\Results\30X5FS\connectedness'+'\\''connected_averaged_over_seeds_3.png')#change this for each case
     d={'p':probs,'epsilon':eps_0,'eps_dev':dev,'p_real':p_r,'p_real_dev':p_r_dev}
     df=pd.DataFrame.from_dict(d)
     df.to_csv(path+r'\videos y fotos medidas\Results\30X5FS\connectedness\connect_30X5FS_3.csv',index=0)
     
#------------------------------------------------------------------------------
#               P_infinite(p) let's see a good behaviour of this thing
#------------------------------------------------------------------------------
def P_inf(sizex,sizey,scale):
      seeds=np.linspace(1,20,20,dtype=int)
      probs=np.linspace(0,1,50)
      pinfs=[]
      probs_real=[]
      preal_dev=[]
      pinfs_dev=[]
      for p in probs:
        p_inf=[]
        preal=[]
        for seed in seeds:
         scipy.random.seed(seed)
         g=UndirectedGraph.MakeRectangularSitePercolation(sizex,sizey,scale,p,seed)
         #g.Fill()
         clusters=g.FindAllClusters()
         total=len(g.GetNodes())
         preal.append(total/(g.L*g.H))
         for cl in clusters:
          if not g.Notpercolates(cl):
              p_inf.append(len(cl)/total)
              pass
          else:
              p_inf.append(0)
              pass
        probs_real.append(np.mean(preal)) 
        preal_dev.append(np.std(preal))
        pinfs.append(np.mean(p_inf))
        pinfs_dev.append(np.std(p_inf))
      df=pd.DataFrame({'p':probs,'p_inf':pinfs,'p_inf_dev':pinfs_dev,'p_real':probs_real,'preal_dev':preal_dev})  
      df.to_csv(path+r'\videos y fotos medidas\Results\30x30NFS\p_inf\data_30x30NFS_Pinf.csv',index=0)
      fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(16, 8))
      ax1.errorbar(probs, pinfs, xerr=0, yerr=pinfs_dev, linestyle='dashed',marker='o',markersize=9)
      ax2.errorbar(probs_real, pinfs, xerr=preal_dev, yerr=pinfs_dev, linestyle='dashed',marker='o',markersize=9)
      ax_params(ax1, xlabel='p',ylabel=r'$p_{\infty}$')
      ax_params(ax2, xlabel='$p_{real}$',ylabel=None)
      fig.savefig(path+r'\videos y fotos medidas\Results\30x30NFS\p_inf\pinf_30x30NFS.png')
#------------------------------------------------------------------------------
#              ns vs s 
#------------------------------------------------------------------------------             
def ns_vs_s(sizex,sizey,scale):
       seeds=np.linspace(1,20,10,dtype=int)
       probs=np.linspace(0,1,10)
       for p in probs:
           d={}
           for seed in seeds:
            scipy.random.seed(seed)
            g=UndirectedGraph.MakeRectangularSitePercolation(sizex,sizey,scale,p,seed)
            g.Fill()
            clusters=g.FindAllClusters()
            n_s=ns(g,clusters)
            keys = np.fromiter(n_s.keys(), dtype=float)
            vals = np.fromiter(n_s.values(), dtype=float)
            keys=np.log(keys)
            vals=np.log(vals)
            plt.plot(keys,vals,ls='dashed',marker='o',label= 'seed=' +str(seed))
            plt.xlabel('s')
            plt.ylabel(r'n_s')
            plt.title('cluster size distribution  vs s  p=%2.2f' %(p))
            plt.legend()
            plt.savefig(path+r'\videos y fotos medidas\Results\30x5FS\log(ns)Vslog(s)'+'\\' 'p=%2.2f' %(p)+'.png')#change this for each case
            d[str(seed)+'_nslog']=keys
            d[str(seed)+'_slog']=vals
           df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))
           df.to_csv(path+r'\videos y fotos medidas\Results\30X5FS\log(ns)Vslog(s)\data_'+str(p)+'.csv',index=0)
           plt.close()
      
#------------------------------------------------------------------------------
#     s vs p   
#------------------------------------------------------------------------------      
def sVSp(sizex,sizey,scale):
        probs=np.linspace(0,1,50)
        seeds=np.linspace(1,20,20,dtype=int)
        totalm=[]
        p_real=[]
        p_real_dev=[]
        for p in probs:
         m=[]
         p_r=[]
         for seed in seeds:
           scipy.random.seed(seed)
           g=UndirectedGraph.MakeRectangularSitePercolation(sizex,sizey,scale,p,seed)
           g.Fill()#added this to the filling case
           clusters=g.FindAllClusters()
           p_r.append(len(g.GetNodes())/(g.H*g.L))
           s=s_sizes(g,clusters)
           keys = np.fromiter(s.keys(), dtype=float)
           if keys.any():
            mean=np.sum(keys)/len(keys)  
            m.append(mean)
           else:
            m.append(0)
         p_real.append(np.mean(p_r))
         p_real_dev.append(np.std(p_r))
         totalm.append(np.mean(m))
        dev=np.std(totalm)
        fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(16,8))
        ax1.errorbar(probs,totalm,xerr=0,yerr=dev, linestyle='dashed',marker='o',markersize=9,color='black',mfc='red')
        ax2.errorbar(p_real,totalm,xerr=p_real_dev,yerr=dev, linestyle='dashed',marker='o',markersize=9,color='black',mfc='red')
        ax_params(ax1,xlabel='p',ylabel=r'\langle S \rangle')
        ax_params(ax2,xlabel=r'$p_{real}$',ylabel=None)
        ax1.set_title(r'cluster size distribution <S> vs p, points= '+str(len(probs)))
        d={'probs':probs,'<s>':totalm,'dev':dev,'p_real':p_real,'p_real_dev':p_real_dev}
        df=pd.DataFrame.from_dict(d)
        df.to_csv(path+r'\videos y fotos medidas\Results\30x5FC\averageSVsP\30X5FC_SvsP.csv', index = False)
        fig.savefig(path+r'\videos y fotos medidas\Results\30X5FC\averageSVsP'+'\\' '30X5FC_points= ' +str(len(probs))+'.png')#change this for each case
        
#------------------------------------------------------------------------------
#     sns vs p   
#------------------------------------------------------------------------------      
def snsVSp(sizex,sizey,scale):
        probs=np.linspace(0,1,20)
        seeds=np.linspace(0,20,20,dtype=int)
        sns=[]
        sns_dev=[]
        preal=[]
        preal_dev=[]
        for p in probs:
          ss=[]
          p_real=[]
          sume=0
          for seed in seeds:
           g=UndirectedGraph.MakeRectangularSitePercolation(sizex,sizey,scale,p,seed)
           clusters=g.FindAllClusters()
           S=s_sizes(g,clusters)
           n_s=ns(g,clusters)
           total=len(g.GetNodes())
           sizetotal=g.L*g.H
           p_real.append(total/sizetotal)
           for s in S.keys():
               sume+=s*n_s[s]
           ss.append(sume)
          sns.append(np.mean(ss))
          sns_dev.append(np.std(ss))
          preal.append(np.mean(p_real))
          preal_dev.append(np.std(p_real))
        data=pd.DataFrame({'p':probs,'p_real':preal,'p_real_dev':preal_dev,'sum_sns':sns,'sum_sns_dev':sns_dev})
        fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(16,8))
        ax1.errorbar(data['p'], data['sum_sns'], xerr=0, yerr=data['sum_sns_dev'], linestyle='dashed',marker='o',markersize=9,color='black',mfc='red')
        ax2.errorbar(data['p_real'], data['sum_sns'], xerr=data['p_real_dev'], yerr=data['sum_sns_dev'], linestyle='dashed',marker='o',markersize=9,color='black',mfc='red')
        ax_params(ax1,xlabel="p",ylabel=r'$\sum Sn_s$')
        ax_params(ax2,xlabel='$p_{real}$',ylabel=None)
        fig.suptitle(r'cluster size distribution $\sum sn_s$ vs p, points= '+str(len(probs)))
        fig.savefig(path+r'\videos y fotos medidas\Results\30x30NFC\sumSnsp\snsp_30x30NFS.png')#change this for each case
        data.to_csv(path+r'\videos y fotos medidas\Results\30x30NFC\sumSnsp\data_sns_30x30NFS.csv', index = False)
        
            
#------------------------------------------------------------------------------
#    Makes linear regression     
#------------------------------------------------------------------------------
def linear(g,X,Y):
     X = X.reshape(-1, 1)  # values converts it into a numpy array
     Y = Y.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column basicamente la transpuesta (1,2,3..) 
     linear_regressor = LinearRegression()  # create object for the class
     linear_regressor.fit(X, Y)  # perform linear regression
     Y_pred = linear_regressor.predict(X)  # make predictions
     plt.plot(X, Y_pred, color='c',ls='dashed',markersize=2)
     plt.plot(X, Y, color='black',ls=' ',marker='1',markersize=3)#esta linea roja si es una regresion
     plt.xlabel(r'$ln(p-p_c)$')
     plt.ylabel(r'$ln(\epsilon(p))$')
     pendiente=linear_regressor.coef_
     inter=linear_regressor.intercept_
     print(' La pendiente : %5.3f' %(pendiente)+' \n El intercepto :  %5.3f' %(inter) )
#------------------------------------------------------------------------------
#                           Graphing Tools
#------------------------------------------------------------------------------

def ax_params(ax,**kwargs):
  ax.set_xlabel(kwargs['xlabel'],fontsize=30)
  if kwargs['ylabel']:
   ax.set_ylabel(kwargs['ylabel'],fontsize=30)
  ax.tick_params( labelsize=20)
    