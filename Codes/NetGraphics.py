import scipy
import os
import random
import tempfile
import shutil
import numpy as np
import math
import cairo
import IPython.display
try:
    import Image
    import ImageDraw
    import ImageFont
except:
    try:
        from PIL import Image, ImageDraw, ImageFont,ImagePath
    except:
        raise Error("PIL package not installed")

# some graphics for networks

# -----------------------------------------------------------------------

# image display


def Display(image_file='tmpf.jpg'):
    """Display(image_file) attempts to display the specified image_file on
    the screen, using the Preview application on Mac OS X, the ImageMagick
    display utility on other posix platforms (e.g., Linux)
    and the Microsoft mspaint utility on Windows platforms.
    """
    os_name = os.name
    if os_name == 'nt':  # Windows
        try:
            os.system('start mspaint %s' % image_file)
        except:
            raise OSError("Cannot display %s with Windows mspaint" %
                          image_file)
    else:
        os_uname = os.uname()
        if os_uname[0] == 'Darwin':  # Mac OS X, assume no X server running
            try:
                os.system('open /Applications/Preview.app %s &' % image_file)
            except:
                raise OSError("Cannot display %s with Preview application" %
                              image_file)

        elif os_name == 'posix':  # Linux, Unix, etc.
            try:
                os.system('display %s &' % image_file)
            except:
                raise OSError("Cannot display %s with ImageMagick display. ImageMagick display requires a running X server." %
                              image_file)
        else:
            raise OSError("no known display function for OS %s" % os_name)



# -----------------------------------------------------------------------

# 2D percolation graphics


def DrawSquareNetworkBonds(graph, nodelists=None,
                           dotsize=None, linewidth=None,
                           imsize=800, windowMargin=0.02, imfile=None):
    """DrawSquareNetworkBonds(g) will draw an image file of the 2D
    square--lattice bond percolation network g, with bonds and sites shown,
    and then will display the result.
    DrawSquareNetworkBonds(g,nodelists) for a percolation graph g and a list of
    node clusters nodelists = [[node,node],[node],...] will draw the
    first cluster black, and the rest each in a random color.
    By default, the image file will be stored in a uniquely named png file
    in /tmp, although the image file name can be supplied optionally with
    the imfile argument.
    A node is a tuple (i,j): DrawSquareNetworkBonds will display it with i
    labeling the horizontal axis and j the vertical, with (0,0) in the
    upper right hand corner. [This is the transpose of the matrix
    convention, so (column, row); it's flipping the vertical axis, so
    it's like (x, -y) ]"""
    # Set up cluster of all nodes in network if no clusters given
    if nodelists is None:
        nodelists = [graph.GetNodes()]
    # Set up image file
    if imfile is None:
        imfile = tempfile.mktemp()  # make unique filename in /tmp
        imfile += "_square_network_bonds.png"
    white = (255, 255, 255)  # background color
    im = Image.new('RGB', (imsize, imsize), color=white)
    draw = ImageDraw.Draw(im)
    # Nodes = (ix, iy) running from (0,0) to (L-1,L-1)
    # Won't always work for site percolation:
    # Assumes entire row and column of nodes not missing
    L = max(max([node[0] for node in graph.GetNodes()]),
            max([node[1] for node in graph.GetNodes()])) + 1.0
    # Default dot size and line width depends on L
    if dotsize is None:
        dotsize = max((1 - 2 * windowMargin) * imsize / (4 * L), 1)
    if linewidth is None:
        linewidth = max((1 - 2 * windowMargin) * imsize / (10 * L), 1)
    # Start colors with black
    color = (0, 0, 0)
    # Draw clusters
    for cluster in nodelists:
        # Define screen location (sx,sy) for node
        # node = (ix, iy) running from (0,0) to (L-1,L-1)
        # Displace on screen to 1/2 ... L-1/2, with margins on each side
        def ScreenPos(i):
            return (windowMargin + ((i + 0.5) / L) *
                    (1 - 2 * windowMargin)) * imsize
        # Find screen location (sx,sy) for node
        for node in cluster:
            ix, iy = node  # node = (ix,iy) running from (0,0) to (L-1,L-1)
            sx = ScreenPos(ix)
            sy = ScreenPos(iy)
            draw.ellipse(((sx - dotsize / 2, sy - dotsize / 2),
                          (sx + dotsize / 2, sy + dotsize / 2)), fill=color)
        # Define function to draw thick line

            def DrawThickLine(sx1, sy1, sx2, sy2):
                perpLength = scipy.sqrt((sy2 - sy1)**2 + (sx2 - sx1)**2)
                perpx = ((sy2 - sy1) / perpLength) * linewidth / 2
                perpy = (-(sx2 - sx1) / perpLength) * linewidth / 2
                polyFromLine = ((sx1 + perpx, sy1 + perpy),
                                (sx1 - perpx, sy1 - perpy),
                                (sx2 - perpx, sy2 - perpy),
                                (sx2 + perpx, sy2 + perpy))
                draw.polygon(polyFromLine, fill=color)
            # Find neighbors
            neighbors = graph.GetNeighbors(node)
            for neighbor in neighbors:
                # Draw each bond once: only if i>j
                if neighbor <= node:
                    continue
                # Find screen location (sxNbr,syNbr)) for edge
                ixNbr, iyNbr = neighbor
                sxNbr = (windowMargin + ((ixNbr + 0.5) / L)
                         * (1 - 2 * windowMargin)) * imsize
                syNbr = (windowMargin + ((iyNbr + 0.5) / L)
                         * (1 - 2 * windowMargin)) * imsize

                # Periodic boundary conditions make this tricky:
                # bonds which cross boundary drawn half a bond length both ways
                # Only nearest neighbor bonds implemented
                if (ix == 0) & (ixNbr == L - 1):
                    sxMinusHalf = ScreenPos(ix - 0.5)
                    DrawThickLine(sx, sy, sxMinusHalf, syNbr)
                    sxNbrPlusHalf = ScreenPos(ixNbr + 0.5)
                    DrawThickLine(sxNbrPlusHalf, sy, sxNbr, syNbr)
                elif (ix == L - 1) & (ixNbr == 0):
                    sxPlusHalf = ScreenPos(ix + 0.5)
                    DrawThickLine(sx, sy, sxPlusHalf, syNbr)
                    sxNbrMinusHalf = ScreenPos(ixNbr - 0.5)
                    DrawThickLine(sxNbrMinusHalf, sy, sxNbr, syNbr)
                elif (iy == 0) & (iyNbr == L - 1):
                    syMinusHalf = ScreenPos(iy - 0.5)
                    DrawThickLine(sx, sy, sxNbr, syMinusHalf)
                    syNbrPlusHalf = ScreenPos(iyNbr + 0.5)
                    DrawThickLine(sx, syNbrPlusHalf, sxNbr, syNbr)
                elif (iy == L - 1) & (iyNbr == 0):
                    syPlusHalf = ScreenPos(iy + 0.5)
                    DrawThickLine(sx, sy, sxNbr, syPlusHalf)
                    syNbrMinusHalf = ScreenPos(iyNbr - 0.5)
                    DrawThickLine(sx, syNbrMinusHalf, sxNbr, syNbr)
                else:
                    DrawThickLine(sx, sy, sxNbr, syNbr)
        # Pick random color for next cluster
        colorRange = (0, 200)
        color = (random.randint(*colorRange),
                 random.randint(*colorRange),
                 random.randint(*colorRange))
    im.save(imfile)
    Display(imfile)
    return im
def convert(a):
    return int(a/ 0.00635)#Resolution,convert cm to pixel from 400 dpi

def DrawHexagonNetworkSites(graph,L,H,p, imsizex,imsizey,nodelists, scale,seed, change=True, imfile=None):
    pixelx=convert(imsizex)
    pixely=convert(imsizey)
    side=6
    hold=convert(5)
    dire1=r'C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp\cortes\Hexagon\color'
    dire2=r'C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp\cortes\Hexagon\black'
    # Background white (in case some nodes missing)
    white = (255, 255, 255)
    color=(0,0,0)
    im = Image.new('RGB', (pixelx, pixely), white)
    if (scale > 1):
        draw = ImageDraw.Draw(im)
    # Draw clusters
    for cluster in nodelists:
        if (scale == 1):
            for node in cluster:
                x=node[0]+hold
                im.putpixel((x,node[1]), color)
        else:
            for node in cluster:
                x = (node[0]*2+1)* scale/2+hold#with this method of drawing, a site would occupy two squares, so when drawing i simply divide the scale by 2,
                y = (node[1]*2+1)* scale/2
                xy = [((math.cos(th)*scale/2 +x) ,(math.sin(th)*scale/2 + y)) for th in [i * (2 * math.pi) / side for i in range(side)]]
                draw.polygon(xy, fill =color)#major change
        # Pick random color for next cluster
        colorRange = (0, 200)
        if change:
         color = (random.randint(*colorRange),
                 random.randint(*colorRange),
                 random.randint(*colorRange))
    x=convert(imsizex-0.5)
    y=convert(imsizey/2)
    r=convert(0.2)
    liste=(x-r, y-r, x+r, y+r)
    draw.ellipse(liste , fill=(0,0,0,0))
    font = ImageFont.truetype(r'C:\Users\Carolina\AppData\Local\Microsoft\Windows\Fonts\AdobeKaitiStd-Regular.otf', 16)
    draw.text((10, 10),'p= ' +str(p),(0,0,0),font=font)
    if change: 
     file=dire1+"\\" + str(p)+str(seed)+'.png'
     im.save(file,'PNG',dpi=(400,400))
    else:
     file2=dire2+"\\" + str(p)+str(seed)+'.png'
     im.save(file2,'PNG',dpi=(400,400))
    #Display(file)
    return im
def DrawCircularNetworkSites(borderup,borderdown,squares,L,H,p, imsizex,imsizey,nodelists, scale,seed, change=True, imfile=None):
    pixelx=convert(imsizex)
    pixely=convert(imsizey)
    hold=convert(9)
    rad=scale/2
    dire1=r'C:\Users\Carolina\OneDrive\Escritorio\color'
    dire2=r'C:\Users\Carolina\OneDrive\Escritorio\black'
    # Background white (in case some nodes missing)
    white = (255, 255, 255)
    color=[0,0,0]
    if change: 
     file=dire1+"\\" + str(p)+str(seed)+'.svg'
    else:
     file=dire2+"\\" + str(p)+str(seed)+'.svg'
    with cairo.SVGSurface(file, pixelx, pixely) as surface:
     context = cairo.Context(surface)
     # Draw clusters
     for cluster in nodelists:
            for node in cluster:
                x = node[0]* scale+hold
                y = node[1] * scale
                context.arc(x+rad, y+rad, rad, 0, 2*math.pi)
                context.set_source_rgb(*color)
                context.fill()
                if squares[node]:
                    context.rectangle(x+rad, y+rad, scale,scale)
                    context.set_source_rgb(*color)
                    context.fill()
                if node[1]==0:
                 if borderup[node]:
                    context.rectangle(x+scale/2, y ,scale, rad)
                    context.set_source_rgb(*color)
                    context.fill()
                if node[1]==49:   
                 if borderdown[node]:
                    context.rectangle(x+scale/2, y+scale/2, scale, rad)
                    context.set_source_rgb(*color)
                    context.fill()
                            
               # Pick random color for next cluster
            if change:
                color = list(np.random.choice(range(100), size=3)/100)
                
     x=scale*305+hold
     dist=0.26
     r=convert(0.125)
     for i in range (9):
            color=[0,0,0]
            context.set_source_rgb(*color)
            y=convert(dist*3/2+dist*2*i)
            context.arc(x, y, r,0, 2*math.pi)
            context.fill()
           
def DrawSquareNetworkSites(L,H,p, imsizex,imsizey,nodelists, scale, seed, change=True):
    pixelx=convert(imsizex)
    pixely=convert(imsizey)
    hold=convert(9)
    dire1=r'C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp\cortes\Rectangular\color\longer'
    dire2=r'C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp\cortes\Rectangular\black\longer'
    # Background white (in case some nodes missing)
    color=[0,0,0]
    if change: 
     file=dire1+"\\" + str(p)+str(seed)+'.svg'
    else:
     file=dire2+"\\" + str(p)+str(seed)+'.svg'
    with cairo.SVGSurface(file, pixelx, pixely) as surface:
     context = cairo.Context(surface)
     for cluster in nodelists:
         context.set_source_rgb(*color)
         for node in cluster:
            x = node[0]* scale+hold
            y = node[1] * scale
            context.rectangle(x, y,  scale, scale)
            context.fill()
         # Pick random color for next cluster
         colorRange = (0, 200)
         if change:
          color = (random.randint(*colorRange),
                  random.randint(*colorRange),
                  random.randint(*colorRange))
     x=scale*305+hold
     dist=0.26
     r=convert(0.125)
     for i in range (9):
      y=convert(dist*3/2+dist*2*i)
      context.arc(x, y, r,0, 2*math.pi)
      context.fill()

# Copyright (C) Cornell University
# All rights reserved.
# Apache License, Version 2.0
#this was partly modified to suit my interests