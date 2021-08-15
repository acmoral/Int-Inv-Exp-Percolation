# Import the necessary libraries.
import NetGraphics
import imp
import numpy as np
import matplotlib.ticker as mtick

import pandas as pd 
#import IntegerHistogram
#import Multiplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(figsize=(16, 12), dpi=144)
#from matplotlib import *
import scipy 
imp.reload(NetGraphics) 
import random
import math
import imp
from sklearn.linear_model import LinearRegression
path=r'C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp'
def ticks(y, pos):
    return r'$e^{:.0f}$'.format(np.log(y))


#-------------------Global constants-------------------------------------------
#------------------------------------------------------------------------------
res= 0.00635 #resolution of image dpi=144
#pylab = MultiPlot.pylab
# -----------------------------------------------------------------------------
#
# Defining undirected graph class: used by both Small World and Percolation
# exercises.
#
# -----------------------------------------------------------------------------

class UndirectedGraph:
# -----------------------------------------------------------------------------
# init method
# -----------------------------------------------------------------------------
    def __init__(self): 
     self.connections,self.circle,self.borderup,self.borderdown={},{},{},{}
     self.counter,self.halfcounter,self.L,self.H=0,0,0,0
# -----------------------------------------------------------------------------
#Checking if node is in graph
# -----------------------------------------------------------------------------
    def HasNode(self, node):
       if node in self.connections.keys():
           return True
       else:
           return False
    def setCounter(self, counter): 
        self.counter=counter
    def setHalfCounter(self, counter): 
        self.halfcounter=counter
    def setLH(self, sizex,sizey,scale): 
        L=math.floor((sizex-6)/(res))
        H=math.floor((sizey)/(res))
        L=int(L/scale)#in pixel
        H=int(H/scale)# in pixel
        self.L=L
        self.H=H
# -----------------------------------------------------------------------------
#Adding node to graph
# -----------------------------------------------------------------------------
    def AddNode(self, node):
        if self.HasNode(node):
            pass
        else:
            self.connections[node] = []
            self.circle[node] = False   
            if node[1]==0:
             self.borderup[node]=False
            elif node[1]==self.H:
             self.borderdown[node] = False
# -----------------------------------------------------------------------------
#Adding edge to graph
# -----------------------------------------------------------------------------
    def AddEdge(self, node1, node2):
        self.AddNode(node1)
        self.AddNode(node2)
        if node2 in self.connections.get(node1):
            pass
        else:
            self.connections.get(node1).append(node2)
    
        if node1 in self.connections.get(node2):
            pass
        else:
            self.connections.get(node2).append(node1)
# -----------------------------------------------------------------------------
#Nodes of graph
# -----------------------------------------------------------------------------
    def GetNodes(self):
        return list(self.connections.keys())
    def GetNodesup(self):
        return list(self.borderup.keys())
    def GetNodesdown(self):
        return list(self.borderdown.keys())
     
# -----------------------------------------------------------------------
#Adding edge random  to graph for percolation exercise
# -----------------------------------------------------------------------
    def AddEdgeRandom(self,po):
     key=self.GetNodes()
     copykey=key.copy()
     p=random.random()
     for i in key:
      for j in copykey:
        if  j  in self.connections.get(i):
         pass
        else:
         if p >=1-po:
          self.connections.get(i).append(j)
        p=random.random()
      copykey.remove(i)
# -----------------------------------------------------------------------
#Adding edge to graph random for Small World
# -----------------------------------------------------------------------
    def AddEdgeRandom2(self,po,seed):
        L=self.L
        H=self.H
        random.seed(seed)
        for i in range(0,L):
         for j in range(0,H):
            if random.random()<=po:
             self.AddNode((i,j))
# -----------------------------------------------------------------------
#Get neightbors is used to find the cluster size later
# -----------------------------------------------------------------------
    def GetNeighbors(self, node):
        return self.connections.get(node)[:]
# -----------------------------------------------------------------------
#For just one node find its cluter
# ------------------------------------------------------------------------
    def FindClusterFromNode(self, node, visited=None):
      if visited is None:
          visited = dict([(n, False) for n in self.GetNodes()])#Creates a dictionary of each node as key and with all values set to False
      cluster = [node]#set current to the node im at
      visited[node] = True#i visited my own node
      currentShell = self.GetNeighbors(node)#the current shell the neighbors
      while len(currentShell) > 0:#if i still have somewhere to flow to, say neighbors, continue
          nextShell = []
          for node in currentShell:
              if not visited[node]:#if it hasnt been checked
                  nextShell.extend(self.GetNeighbors(node))#get the neighbors of neighbor at the end of list
                  visited[node] = True#check it
                  cluster.append(node)#add to cluster
          currentShell = nextShell
      return cluster
# -----------------------------------------------------------------------
#fins all clusters
# ------------------------------------------------------------------------
    def FindAllClusters(self):
     visited = dict([(n, False) for n in self.GetNodes()])
     clusters = []
     for node in self.GetNodes():
         if not visited[node]:
             cluster = self.FindClusterFromNode(node, visited)#this visited dict is modified in FindClustedfromNode
             clusters.append(cluster)
            #clusters.sort(lambda x,y: cmp(len(y), len(x))) # reverse sort of len()

     clusters.sort(key=len, reverse=True)  # reverse sort of len() 
     return clusters
# -----------------------------------------------------------------------
#This could be used too, for square percolation
# ------------------------------------------------------------------------
    def MakeSquareBondPercolation(p,seed):
     g=UndirectedGraph()
     random.seed(seed)
     L=g.L
     H=g.H
     for i in range(0,L):
         for j in range(0,H): 
           g.AddNode((i,j))
     for l in range(0,L):
      for k in range(0,H):
       horizontal=random.random()
       vertical=random.random()
       if horizontal<p:
        g.AddEdge((l,k),(l,(k+1)%H))
       if vertical <p:
        g.AddEdge((l,k),((l+1)%L,k))
     return g
# -----------------------------------------------------------------------
#This draws the occupation histograms
# ------------------------------------------------------------------------
    def drawhist(self,rect,p,bars,sizex,sizey):
      r=0.05
      L=self.L
      H=self.H
      cl = self.FindAllClusters()
      h=[len(l) for l in cl]
      h1=self.GetSizeDistribution(cl)
      x=list(h1.keys())
      y=h1.values()
      y=list(y)
      y.reverse()
      x.sort()
      fig, ax = plt.subplots(figsize=(10, 10))
      if bars==True:
       if p>0.55: #I changed this for the circular case because it gets messy at this point
          names=[str(i) for i in x]
          ax.set_xticklabels(names,rotation=45,fontsize = 10.0)
          ax.bar(names,y)
       else:
          ax.hist(h)
      if rect:
          p_real=len(self.GetNodes())/(L*H)
          p_otro=p
      else:
          circle=math.pi*(r**2)
          square=4*(r**2)
          nodesin=len(self.GetNodes())
          Area=square-circle
          Areah=Area/2
          ocupated=nodesin*circle+(self.counter*Area)+(self.halfcounter*Areah)
          p_real=ocupated/(sizex*sizey) #real occupated area
          p_otro=p
      ax.set_xlabel('Tamaño de Cluster', fontsize = 20.0)
      ax.set_ylabel('frecuencia', fontsize = 20.0)
      ax.annotate(' L=%3d \n H= %3d \n O=%3d' %(L,H,len(self.GetNodes())),xy=(0.9, 0.9),xycoords='axes fraction',horizontalalignment='right', verticalalignment='top',fontsize=16)
      ax.set_title(r'$p_{deseado}= $'+ str(p_otro)+'  '+r' $p_{real}=$'+" %5.5f" % (p_real), fontsize = 20.0)
      ax.figure.savefig(path+r'\cortes\Rectangular\plot\ideal'+'\\' +str(p)+'.png')#change this for each case
# -----------------------------------------------------------------------
#This is used to generate the rectangular percolation
# ------------------------------------------------------------------------    
    def PlotCircular(p, seed, sizex,sizey,scale,bars=True):
     scipy.random.seed(seed)
     g=UndirectedGraph.MakeRectangularSitePercolation(sizex,sizey,scale,p,seed)
     g.Fill()
     g.Circlefiller()
     g.fillborder()
     cl = g.FindAllClusters()
     squares=g.circle
     borderup=g.borderup
     borderdown=g.borderdown
     L=g.L
     H=g.H
     NetGraphics.DrawCircularNetworkSites(borderup,borderdown,squares,L,H,p,nodelists=cl,imsizex=sizex,imsizey=sizey,scale=scale,change=False)
     NetGraphics.DrawCircularNetworkSites(borderup,borderdown,squares,L,H,p,nodelists=cl,imsizex=sizex,imsizey=sizey,scale=scale,change=True)
     g.drawhist(False,p,bars,sizex,sizey)
# -----------------------------------------------------------------------
#This plots the actual pattern generated in the material
# ------------------------------------------------------------------------    
    def PlotRectReal(p, seed, sizex,sizey,scale,bars=True):
     scipy.random.seed(seed)
     g=UndirectedGraph.MakeRectangularSitePercolation(sizex,sizey,scale, p,seed)
     L=g.L
     H=g.H
     g.Fill()#fill it
     cl = g.FindAllClusters()
     NetGraphics.DrawSquareNetworkSites(L,H,p,nodelists=cl,imsizex=sizex,imsizey=sizey,scale=scale,change=False)
     NetGraphics.DrawSquareNetworkSites(L,H,p,nodelists=cl,imsizex=sizex,imsizey=sizey,scale=scale,change=True)
     g.drawhist(True,p,bars,sizex,sizey)
# -----------------------------------------------------------------------
#This returns the actual printed probability, it could be used to compare with the one that is required
# ------------------------------------------------------------------------    
    def PvsP(p, seed, sizex,sizey,scale,bars=True):
     scipy.random.seed(seed)
     g=UndirectedGraph.MakeRectangularSitePercolation(sizex,sizey,scale, p,seed)
     g.Fill()
     L=g.L
     H=g.H
     sume=len(g.GetNodes())/(L*H)
     return sume
# -----------------------------------------------------------------------
#Returns a dictionary contaning the number of clusters given a size of cluster
# ------------------------------------------------------------------------     
    def GetSizeDistribution(self,clusters):
     #sizes = [len(cl) for cl in clusters]
     #max_size = max(sizes)
     hist = {}
     for cl in clusters:
         if len(cl) in hist:
             hist[len(cl)] += 1
         else:
             hist[len(cl)] = 1
     return hist
# -----------------------------------------------------------------------
#Makes the graph for rectangular percolation
# ------------------------------------------------------------------------ 
    def MakeRectangularSitePercolation(sizex,sizey,scale,p,seed):
      g =UndirectedGraph()
      g.setLH(sizex, sizey,scale)
      g.AddEdgeRandom2(p,seed)
      g.AddN()
      return g
# -----------------------------------------------------------------------
#This fills the holes that fall naturally in the real world but not in the image
# ------------------------------------------------------------------------  
    def Fill(self):
      H=self.H
      inv=self.Invert()
      cl=inv.FindAllClusters()
      c=0
      verify=[(0,j) for j in range(H)]
      for cluster in cl:
        for n in verify :
            if n in cluster:
                c+=1
                pass
        if c==0:  
         for obj in cluster:
          self.AddNode(obj)
        c=0
      self.AddN()
# -----------------------------------------------------------------------
#In order to find the holes to fill we invert the image first, then verify if they are connected to the left side of the material
# ------------------------------------------------------------------------  
    def Invert(self):
       L=self.L
       H=self.H
       ginv=UndirectedGraph()
       ginv.setLH(L, H, 15.74)
       nodes= self.GetNodes()
       for i in range(0,L):
           for j in range(0,H):
               if (i,j) in nodes:
                   nodes.remove((i,j))
                   pass
               else:
                  ginv.AddNode((i,j))
       ginv.AddN2()
       return ginv
# -----------------------------------------------------------------------
#Adds neighbors given the existing nodes, it goes by the lattice seeing who surrounds whom
# ------------------------------------------------------------------------  
    def AddN(self):
      L=self.L
      H=self.H
      nodes=self.GetNodes()
      nbrs =[[0, 1], [1, 0] ,[-1,0],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]] 
      for node in nodes:
                  for nbr in nbrs:
                     if (node[0]+nbr[0])>L or (node[1]+nbr[1])>H or (node[0]+nbr[0])<0 or  (node[1]+nbr[1])<0:
                         pass
                     else:
                         site = (node[0] + nbr[0], node[1] + nbr[1])
                         if self.HasNode(site):
                           self.AddEdge(node, site)
# -----------------------------------------------------------------------
#But it is not the same for the inverted clusters, if the holes are connected diagonally(which theya re in real life) then materials cannot be connected diagonally
#that is, either the inverted cluster has diagonal neighbors or the other
#in the case for circular i said that in the normal non inverted cluster the diagonal dots are not neighbors, because they occupy less space, that means that non dot spaces are connected in dual space
# ------------------------------------------------------------------------   
    def AddN2(self):
      L=self.L
      H=self.H
      nodes=self.GetNodes()
      nbrs = [[0, 1], [1, 0] ,[-1,0],[0,-1]]  #this was changed from rectangular to circular
      for node in nodes:
                  for nbr in nbrs:
                     if (node[0]+nbr[0])>L or (node[1]+nbr[1])>H or (node[0]+nbr[0])<0 or  (node[1]+nbr[1])<0:
                         pass
                     else:
                         site = (node[0] + nbr[0], node[1] + nbr[1])
                         if self.HasNode(site):
                           self.AddEdge(node, site)
#..................Verify percolation pc.........................................
    def PlotPc(p, seed, sizex,sizey,scale):
     scipy.random.seed(seed)
     g=UndirectedGraph.MakeRectangularSitePercolation(sizex,sizey, p,seed)
     g.Fill()#fill it with naturally falling sites
     cl = g.FindAllClusters()
     L=g.L
     H=g.H
     NetGraphics.DrawSquareNetworkSites(g,L,H,p,nodelists=cl,imsizex=sizex,imsizey=sizey,scale=scale,change=True)
#----------------Gyration Radious-------------------------------------------
    def rmean(self,cl):
     S=len(cl)
     rmean=np.sum(cl,axis=0)/S
     return rmean
#-------------------------------------------------------------------------------
    def Gradious(self,cl):
      mean=self.rmean(cl)
      R_i=0
      for i in cl:
        value=i-mean
        R_i+=np.dot(value,value)
      return R_i
#------------------------------------------------------------------------------
#               verify if percolates, either horizontally or vertically
#------------------------------------------------------------------------------
    def Notpercolates(self,cl):
     L=self.L
     H=self.H
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
          if not g.Notpercolates(cl):
              return True,preal
     return False,preal
              
 #------------------------------------------------------------------------------
#               Divide the clusters according to size
#------------------------------------------------------------------------------         
    def s_sizes(self,clusters):
      d={} # an empty dict that is going to contain keys:sizes and values: the index in clusters list
      i=0
      for cl in clusters:
         lenght=len(cl)
         if self.Notpercolates(cl):
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
    def ns(self,clusters):
      d=self.s_sizes(clusters)#dictionary of sizes
      s={}#another dict for the cluster size distribution
      L=self.L
      H=self.H
      for key in d.keys():
       s[key]=len(d[key])/(L*H)
      return s
#-------------------------------------------------------------------------------
    def Rs_average(self,clusters):
     s=self.s_sizes(clusters)
     Rs={}
     for key in s.keys():
         if key!=0:
          numbers=s[key]
          ss=0
          for i in numbers:
            ss+=self.Gradious(clusters[i])/key
          Rs[key]=ss/len(numbers)
         else:
             Rs[key]=0
     return Rs
#-------------------------------------------------------------------------------
    def eps(self,clusters):
     s=self.s_sizes(clusters)
     n_s=self.ns(clusters)
     R_s=self.Rs_average(clusters)
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
     ns=g.ns(clusters)
     S=g.s_sizes(clusters)
     sume=0
     for s in S.keys():
         sume+=s*ns[s]
     if sume*N==total:
         print(sume*N,total)
         return True
     else:
         print(sume*N,total)
         return False
        
#-------------------------------------------------------------------------------
    def connect(sizex,sizey,scale,pc):
     seeds=np.linspace(1,20,20,dtype=int)
     eps=[]
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
         epsilon=g.eps(clusters)
         Y.append(epsilon)
         p_real.append(len(g.GetNodes())/(g.L*g.H))
       eps.append(np.mean(Y))
       dev.append(np.std(Y))
       p_r.append(np.mean(p_real))
       p_r_dev.append(np.std(p_real))
     fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(16,8))
     ax1.plot(probs,eps,marker='D',linestyle=' ',markersize=6,markerfacecolor="None")
     ax2.errorbar(p_r,eps,xerr=p_r_dev,yerr=dev,marker='D',linestyle=' ',markersize=6,markerfacecolor="None")
     ax1.yaxis.set_major_formatter(mtick.FuncFormatter(ticks))
     ax1.xaxis.set_major_formatter(mtick.FuncFormatter(ticks)) 
     ax2.yaxis.set_major_formatter(mtick.FuncFormatter(ticks))
     ax2.xaxis.set_major_formatter(mtick.FuncFormatter(ticks)) 
     ax1.set_xlabel('p',fontsize=30)
     ax2.set_xlabel('$p_{real}$',fontsize=30)
     ax1.set_ylabel(r'$\xi(p)$',fontsize=30)
     ax1.tick_params( labelsize=20)
     ax2.tick_params( labelsize=20)
     #ax.set_xlabel('log(p)')
     #ax.set_ylabel(r'$log(\epsilon(p))$')
     fig.savefig(path+r'\videos y fotos medidas\Results\30X5FS\connectedness'+'\\''connected_averaged_over_seeds_2.png')#change this for each case
     d={'p':probs,'epsilon':eps,'eps_dev':dev,'p_real':p_r,'p_real_dev':p_r_dev}
     df=pd.DataFrame.from_dict(d)
     df.to_csv(path+r'\videos y fotos medidas\Results\30X5FS\connectedness\connect_30X5FS_2.csv',index=0)
     
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
      fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(16, 8))#aca puedes cambiar el tamaño de la figura 12 es el ancho y 8 la altura
      ax1.errorbar(probs, pinfs, xerr=0, yerr=pinfs_dev, linestyle='dashed',marker='o',markersize=9)
      ax2.errorbar(probs_real, pinfs, xerr=preal_dev, yerr=pinfs_dev, linestyle='dashed',marker='o',markersize=9)
      ax1.set_xlabel('p',fontsize=30)
      ax2.set_xlabel('$p_{real}$',fontsize=30)
      ax1.set_ylabel(r'$p_{\infty}$',fontsize=30)
      ax1.tick_params( labelsize=20)
      ax2.tick_params( labelsize=20)
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
            n_s=g.ns(clusters)
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
           s=g.s_sizes(clusters)
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
        ax1.set_xlabel('p',fontsize=30)
        ax1.set_ylabel(r'<S>',fontsize=30)
        ax1.tick_params( labelsize=20)
        ax2.tick_params( labelsize=20)
        ax2.set_xlabel(r'$p_{real}$',fontsize=30)
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
           S=g.s_sizes(clusters)
           n_s=g.ns(clusters)
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
        ax1.set_xlabel('p',fontsize=30)
        ax2.set_xlabel('$p_{real}$',fontsize=30)
        ax1.set_ylabel(r'$\sum Sn_s$',fontsize=30)
        ax1.tick_params( labelsize=20)
        ax2.tick_params( labelsize=20)
        fig.suptitle(r'cluster size distribution $\sum sn_s$ vs p, points= '+str(len(probs)))
        fig.savefig(path+r'\videos y fotos medidas\Results\30x30NFC\sumSnsp\snsp_30x30NFS.png')#change this for each case
        data.to_csv(path+r'\videos y fotos medidas\Results\30x30NFC\sumSnsp\data_sns_30x30NFS.csv', index = False)
        
            
#------------------------------------------------------------------------------
#    Makes linear regression     
#------------------------------------------------------------------------------
    def linear(self,X,Y):
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
# -----------------------------------------------------------------------
#Circles are complicated, they have white speaces when they form a four circle group
# -----------------------------------------------------------------------
    def Circlefiller(self):
        nodes=self.GetNodes()
        counter=0
        for node in nodes:
            c=0
            nbrs=self.GetNeighbors(node)
            four=[(node[0]+1,node[1]),(node[0],node[1]+1),(node[0]+1,node[1]+1)]
            for el in four:
                if el in nbrs:
                    c+=1
            if c==3:
             self.circle[node]= True  
             counter+=1
        self.setCounter(counter)
# -----------------------------------------------------------------------
#Also in the border
# -----------------------------------------------------------------------       
    def fillborder(self):
        nodesup=self.GetNodesup()
        nodesdown=self.GetNodesdown()
        countborder=0
        for nodeup in nodesup:
          if (nodeup[0]+1,nodeup[1]) in nodesup:
             countborder+=1
             self.borderup[nodeup]= True
        for noded in nodesdown:
          if (noded[0]+1,noded[1]) in nodesdown:
             countborder+=1
             self.borderdown[noded]= True
        self.setHalfCounter(countborder)

# -----------------------------------------------------------------------
#what is the real probability
# -----------------------------------------------------------------------       
    def preal_plt(sizex,sizey,scale):
     probs=np.linspace(0,1,20)
     seeds=np.linspace(1,20,20,dtype=int)
     p_real=[]
     p_real_dev=[]
     for p in probs:
         p_r=[]
         for seed in seeds:
             g=UndirectedGraph.MakeRectangularSitePercolation(sizex,sizey,scale,p,seed)
             g.Fill()
             p_r.append(len(g.GetNodes())/(g.L*g.H) )
         p_real.append(np.mean(p_r))
         p_real_dev.append(np.std(p_r))
     df=pd.DataFrame({'p_real':p_real,'p_real_dev':p_real_dev,'p_teo':probs})
     df.to_csv(r'C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp\videos y fotos medidas\Results\30X5FS\pvsp_real.csv', index = False)
     fig, (ax1) = plt.subplots(1, 1,figsize=(12, 8))#aca puedes cambiar el tamaño de la figura 12 es el ancho y 8 la altura
     ax1.errorbar(probs, p_real, xerr=0, yerr=p_real_dev, linestyle='dashed',marker='o',markersize=1)
     ax1.set_xlabel('p')
     ax1.set_ylabel(r'$p_real$')
     fig.suptitle(r'p vs $p_real$', fontsize=16)
     fig.savefig(path+r'\videos y fotos medidas\Results\30X5FS'+'\\''pvsp_real.png')#change this for each case
