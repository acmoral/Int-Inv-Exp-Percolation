# Import the necessary libraries.
import NetGraphics
import imp
import numpy as np
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
     self.circular=False
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
        L=math.floor((sizex-10)/(res))
        H=math.floor((sizey)/(res))
        L=int(L/scale)#in pixel
        H=int(H/scale)# in pixel
        self.L=L
        self.H=H
    def set_circular(self):
        self.circular=True
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
            elif node[1]==self.H-1:
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
     g=UndirectedGraph.MakeRectangularSitePercolation(sizex,sizey,scale,p,seed,True)
     g.Fill()
     g.Circlefiller()
     g.fillborder()
     cl = g.FindAllClusters()
     squares=g.circle
     borderup=g.borderup
     borderdown=g.borderdown
     L=g.L
     H=g.H
     NetGraphics.DrawCircularNetworkSites(borderup,borderdown,squares,L,H,p,nodelists=cl,imsizex=sizex,imsizey=sizey,scale=scale,seed=seed,change=False)
     NetGraphics.DrawCircularNetworkSites(borderup,borderdown,squares,L,H,p,nodelists=cl,imsizex=sizex,imsizey=sizey,scale=scale,seed=seed,change=True)
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
     NetGraphics.DrawSquareNetworkSites(L,H,p,sizex,sizey,cl,scale,seed,change=False)
     NetGraphics.DrawSquareNetworkSites(L,H,p,sizex,sizey,cl,scale,seed,change=True)
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
    def MakeRectangularSitePercolation(sizex,sizey,scale,p,seed,circular):
      g =UndirectedGraph()
      if circular:
          g.set_circular()
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
      if self.circular:
         nbrs =[[0, 1], [1, 0] ,[-1,0],[0,-1]]
      else: 
         nbrs = [[0, 1], [1, 0] ,[-1,0],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]] 
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
      if self.circular:
         nbrs =[[0, 1], [1, 0] ,[-1,0],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]] 
      else: 
         nbrs = [[0, 1], [1, 0] ,[-1,0],[0,-1]]
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
# -----------------------------------------------------------------------
#Circles are complicated, they have white speaces when they form a four circle group
# -----------------------------------------------------------------------
    def Circlefiller(self):
        nodes=self.GetNodes()
        counter=0
        for node in nodes:
            nbrs=self.GetNeighbors(node)
            if (node[0],node[1]+1) in nbrs:
                if (node[0]+1,node[1]) in nbrs:
                    if (node[0]+1,node[1]+1) in nodes:
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
