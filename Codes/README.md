# Codes used to generate the percolating patterns
The code used here is based on a problem found in Sethna's book  of statistical mechanics "Entropy, Order Parameters, and Complexity" the problem consists of generating percolating 
patterns in a square LxL lattice, this code is later modified to also generate triangular percolation patterns. I used this code as base and modified it to:
* Generate rectangular patterns in an LxH lattice
* I removed the periodic boundary conditions because it makes no sense in the real world problem i'm dealing with
* I removed holes that in theory are there but not in real printed patterns, that is , in the generated image they are not occupied but in reality they fall because there is no near material to hold up with.

The code is divided as follows: the file NetGraphics is used to generate the image from the nodes and nightbors data of the graph generated. the file Graph.py generated the graphs and 
percolating clusters, finally the file percolation.py is the main file where all is put together to use
