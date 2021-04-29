
# Import the necessary libraries.
import NetGraphics
import imp
import random
import numpy as np
import pandas as pd
import IntegerHistogram
import Multiplot
import matplotlib.pyplot as plt
from matplotlib import *
import scipy as sp
imp.reload(NetGraphics) 
import random, scipy, os, time
import math
import imp
res= 0.00635
#pylab = MultiPlot.pylab
# -----------------------------------------------------------------------
#
# Defining undirected graph class: used by both Small World and Percolation
# exercises.
#
# -----------------------------------------------------------------------

class UndirectedGraph:
# -----------------------------------------------------------------------
# init method
# -----------------------------------------------------------------------
    def __init__(self): 
     self.connections = {}
# -----------------------------------------------------------------------
#Checking if node is in graph
# -----------------------------------------------------------------------
    def HasNode(self, node):
       if node in self.connections.keys():
           return True
       else:
           return False
# -----------------------------------------------------------------------
#Adding node to graph
# -----------------------------------------------------------------------
    def AddNode(self, node):
        if self.HasNode(node):
            pass
        else:
	        self.connections[node] = []
# -----------------------------------------------------------------------
#Adding edge to graph
# -----------------------------------------------------------------------
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
# -----------------------------------------------------------------------
#Nodes of graph
# -----------------------------------------------------------------------
    def GetNodes(self):
        return list(self.connections.keys())
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
    def AddEdgeRandom2(self,po,L,H,seed):
        number=int(po*L*H)
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
    def MakeSquareBondPercolation(L,H, p,seed):
     g=UndirectedGraph()
     random.seed(seed)
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
    def drawhist(self,p,L,H,bars):
      cl = self.FindAllClusters()
      h=[len(l) for l in cl]
      sume=len(self.GetNodes())/(L*H)
      h1=self.GetSizeDistribution()
      x=list(h1.keys())
      y=h1.values()
      y=list(y)
      y.reverse()
      x.sort()
      fig, ax = plt.subplots(figsize=(10, 10))
      if bars==True:
       if p>0.5:
          names=[str(i) for i in x]
          ax.set_xticklabels(names,rotation=45,fontsize = 15.0)
          ax.bar(names,y)
       else:
          ax.hist(h)
      ax.set_xlabel('Tamaño de Cluster', fontsize = 20.0)
      ax.set_ylabel('frecuencia', fontsize = 20.0)
      ax.annotate(' L=%3d \n H= %3d \n O=%3d' %(L,H,sume*L*H),xy=(0.9, 0.9),xycoords='axes fraction',horizontalalignment='right', verticalalignment='top',fontsize=16)
      ax.set_title(r'$p_{deseado}= $'+ str(p)+'  '+r' $p_{real}=$'+" %5.5f" % (sume), fontsize = 20.0)
# -----------------------------------------------------------------------
#This is used to generate the rectangular percolation
# ------------------------------------------------------------------------    
    def PlotRectangular(p, seed, sizex,sizey,scale,bars=True):
     scipy.random.seed(seed)
     L=math.floor((sizex-6)/(res))
     H=math.floor((sizey)/(res))
     L=int(L/scale)
     H=int(H/scale)     
     g=UndirectedGraph.MakeRectangularSitePercolation(L,H, p,seed)
     cl = g.FindAllClusters()
     NetGraphics.DrawSquareNetworkSites(g,L,H,p,nodelists=cl,imsizex=sizex,imsizey=sizey,scale=scale,change=False)
     NetGraphics.DrawSquareNetworkSites(g,L,H,p,nodelists=cl,imsizex=sizex,imsizey=sizey,scale=scale,change=True)
     g.drawhist(p,L,H,bars)
# -----------------------------------------------------------------------
#This plots the actual pattern generated in the material
# ------------------------------------------------------------------------    
    def PlotRectReal(p, seed, sizex,sizey,scale,bars=True):
     scipy.random.seed(seed)
     L=math.floor((sizex-6)/(res))
     H=math.floor((sizey)/(res))
     print(L,H)
     L=int(L/scale)
     H=int(H/scale) 
     print(L,H)
     g=UndirectedGraph.MakeRectangularSitePercolation(L,H, p,seed)
     g.Fill(L,H)#fill it
     cl = g.FindAllClusters()
     NetGraphics.DrawSquareNetworkSites(g,L,H,p,nodelists=cl,imsizex=sizex,imsizey=sizey,scale=scale,change=False)
     NetGraphics.DrawSquareNetworkSites(g,L,H,p,nodelists=cl,imsizex=sizex,imsizey=sizey,scale=scale,change=True)
     g.drawhist(p,L,H,bars)
# -----------------------------------------------------------------------
#This returns the actual printed probability, it could be used to compare with the one that is required
# ------------------------------------------------------------------------    
    def PvsP(p, seed, sizex,sizey,scale,bars=True):
     scipy.random.seed(seed)
     L=math.floor((sizex-6)/(res))
     H=math.floor((sizey)/(res))
     L=int(L/scale)
     H=int(H/scale) 
     g=UndirectedGraph.MakeRectangularSitePercolation(L,H, p,seed)
     g.Fill(L,H)
     cl = g.FindAllClusters()
     sume=len(g.GetNodes())/(L*H)
     return sume
# -----------------------------------------------------------------------
#Returns a dictionary contaning the number of clusters given a size of cluster
# ------------------------------------------------------------------------     
    def GetSizeDistribution(self):
     #sizes = [len(cl) for cl in clusters]
     #max_size = max(sizes)
     hist = {}
     clusters=self.FindAllClusters()
     for cl in clusters:
         if len(cl) in hist:
             hist[len(cl)] += 1
         else:
             hist[len(cl)] = 1
     return hist
# -----------------------------------------------------------------------
#Makes the graph for rectangular percolation
# ------------------------------------------------------------------------ 
    def MakeRectangularSitePercolation(L,H,p,seed):
      g =UndirectedGraph()
      g.AddEdgeRandom2(p,L,H,seed)
      g.AddN(L,H)
      return g
# -----------------------------------------------------------------------
#This fills the holes that fall naturally in the real world but not in the image
# ------------------------------------------------------------------------  
    def Fill(self,L,H):
      inv=self.Invert(L,H)
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
      self.AddN(L,H)
# -----------------------------------------------------------------------
#In order to find the holes to fill we invert the image first, then verify if they are connected to the left side of the material
# ------------------------------------------------------------------------  
    def Invert(self,L,H):
       ginv=UndirectedGraph()
       nodes= self.GetNodes()
       for i in range(0,L):
           for j in range(0,H):
               if (i,j) in nodes:
                   pass
               else:
                  ginv.AddNode((i,j))
       ginv.AddN2(L,H)
       return ginv
# -----------------------------------------------------------------------
#Adds neighbors given the existing nodes, it goes by the lattice seeing who surrounds whom
# ------------------------------------------------------------------------  
    def AddN(self,L,H):
      nodes=self.GetNodes()
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
#that is, either the inverted cluster has diagonal neighbors
# ------------------------------------------------------------------------   
    def AddN2(self,L,H):
      nodes=self.GetNodes()
      nbrs = [[0, 1], [1, 0] ,[-1,0],[0,-1]]
      for node in nodes:
                  for nbr in nbrs:
                     if (node[0]+nbr[0])>L or (node[1]+nbr[1])>H or (node[0]+nbr[0])<0 or  (node[1]+nbr[1])<0:
                         pass
                     else:
                         site = (node[0] + nbr[0], node[1] + nbr[1])
                         if self.HasNode(site):
                           self.AddEdge(node, site)