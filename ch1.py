"""
1. The Power of Linear Algebra


To demonstrate, this chapter introduces vectors, matrices, and matrix-vector
multiplication.

We’ll use these tools to simulate a random walk in a network, and we’ll implement the
PageRank algorithm Google used to revolutionize web search. Along the way, we’ll learn
about eigenvectors, one of the most powerful ideas in linear algebra.
"""
import os.path

from colorama import Fore, Style
from functions import LinearAlgebra
import networkx as nx
import numpy as np
import pandas as pd
import sys
import tarfile
from utils import decorate
from urllib.parse import unquote


# Define colors
red = Fore.LIGHTRED_EX
gold = Fore.YELLOW
yellow = Fore.LIGHTYELLOW_EX
green = Fore.GREEN
cyan = Fore.LIGHTCYAN_EX
blue = Fore.BLUE
pink = Fore.LIGHTMAGENTA_EX
magenta = Fore.MAGENTA
white = Fore.LIGHTWHITE_EX
grey = Fore.WHITE
bold = Style.BRIGHT
rst = Style.RESET_ALL

# Set options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:,.3f}'.format)

roundVal = 2

"""
Notes:

A = [[3, 1],          vals = [4, 2]
     [1, 3]]          vecs = [[1,  1],
                              [1, -1]]

* Eigenvalues and Eigenvectors
    Generally, "x" is an eigenvector of "M" if Mx = λx,
        where λ is an eigenvalue of "M".
    
    The equation uses the following operators:
        M @ x = λ * x


* Matrix Operations:
    Operation	 Shapes	        Result	What it computes
    A @ vecs	 (2,2) @ (2,2)	(2,2)	Each column of V transformed by A
    vals * vecs	 (2,) × (2,2)	(2,2)	Each column of V scaled by its λ (broadcasting)
    vals @ vecs	 (2,) @ (2,2)	(2,)	Weighted column-sum (unrelated)

    A @ vecs == vals * vecs


* Operator: @
    Matrix Multiplication:
    - Vector times vector
        Multiply row_A by col_vecs, then sum
        A @ vecs =
            result[0,0] = 3×1 + 1×1  = 4
            result[0,1] = 3×1 + 1×(-1) = 2
            result[1,0] = 1×1 + 3×1  = 4
            result[1,1] = 1×1 + 3×(-1) = -2
            = [[4,  2],
               [4, -2]]

        Operation	 Shapes	        Result
        A @ vecs	 (2,2) @ (2,2)	(2,2)
    
    Dot Product:
    - Scalar times vector
        vals @ vecs =
            result[0] = [4,2] · col0 of V = 4×1 + 2×1  = 6
            result[1] = [4,2] · col1 of V = 4×1 + 2×(-1) = 2
            = [6, 2]
        
        Operation	 Shapes	        Result
        vals @ vecs	 (2,) @ (2,2)	(2,)


* Operator: *
    Each column scaled independently. No mixing.
    
    - A * vecs
              col0    col1
        row0  4×1     2×1
        row1  4×1     2×(-1)
        
        = [[4,  2],
           [4, -2]]
"""


# Initialize class
la = LinearAlgebra()


"""
1.1. A Day at the Museum:


Suppose you are meeting a friend at the Isabella Stewart Gardner Museum in Boston. You 
have not agreed on a meeting place – and let’s assume your cell phone is dead. If you are 
familiar with the layout of the museum, you might be able to make an intelligent guess 
about where to look for your friend.

As a rough model of the situation, suppose your friend has been at the museum for some 
time, wandering from room to room at random. By simulating this random walk, we can 
compute their probability of being in each room.

As a small example, we’ll create a graph that represents just a few of the rooms and the 
connections between them (based on the actual layout of the Gardner Museum).
"""
inPlotGraph = False
inEdges = [
    ("Lobby", "Living Room"),
    ("Lobby", "Shop"),
    ("Lobby", "Main Stair"),
    ("Living Room", "Main Stair"),
    ("Main Stair", "Cafe"),
    ("Cafe", "Garden"),
]
graph = la.plotGraph(
    edges=inEdges, title='Museum', figName='ch1_graph', nodeSize=1200, plotFig=inPlotGraph
)
rooms = list(graph.nodes)
"""
Random walk:

Now we will simulate a random walk through the museum. 
We'll use a Counter to keep track of the number of times each room is visited. 
Then divide through by the number of iterations, so the values in the Counter sum to one.
"""
la.randomWalk(G=graph, start='Lobby', pHeader=True, pSpace=True)


"""
1.2 Adjacency Matrix:


Each element is either 1 to indicate that there is an edge between two nodes or 0 to 
indicate that there is not.
"""
adjacencyMatrix = la.adjacencyMatrix(G=graph, pHeader=True)
"""
Transition Matrix:

We can use the adjacency matrix to compute a transition matrix, which represents the 
probability that a random walker moves from each room to the others.

Each row is a 1-D array, like a vector.

We can interpret the elements of this vector as transition probabilities from other rooms 
to the Lobby. 
For example, the element corresponding to the Living Room is 0.5, which means that if 
someone is in the Living Room and they follow a random edge, the probability they end up 
in the Lobby is 0.5. 
If they leave the Shop, the probability is 1, and if they leave the Main Stair, the 
probability is 0.333.
"""
transitionMatrix = la.transitionMatrix(
    matrix=adjacencyMatrix, idx=rooms, col=rooms
)
la.showVector(array=transitionMatrix, vector='Lobby', label2='Lobby',
              idx=rooms, col=rooms, pSpace=True)


"""
1.3 Dot Product:


Now suppose there are 100 people in every room. We can create an array that represents 
this hypothetical starting condition.
"""
N = len(graph)
x0 = np.full(N, 100)
la.showVector(array=[x0], idx='Start Condition', col=rooms,
              label='Museum Migration (x0)', pHeader=True)
"""
And now suppose that everyone leaves the room they are in, choosing one of the exits at 
random. To find out how many of them end up in the Lobby, we can multiply row0 by x
"""
selRoom = 'Lobby'
selRoomIdx = rooms.index(selRoom)
row0 = transitionMatrix[selRoomIdx]
product = x0 * row0 # Multiply arrays
nMigrants = round(product.sum(), roundVal)
la.showVector(array=[product], idx=[selRoom], col=rooms,
              label=f'x0 * {selRoom}')
print(f'* A total of {red}{nMigrants:,}{rst} people have moved to the {selRoom}')
"""
The sum of elementwise products is an “inner product”, more often called a dot product 
because the operator used to denote it is a dot.
"""
dotProd = row0.dot(x0)
print(f'* Dot Product: row0.dot(x0) = {red}{round(dotProd, roundVal):,}{rst}')
dotProd = row0 @ x0
print(f'* Dot Product: row0 @ x0 = {red}{round(dotProd, roundVal):,}{rst}\n\n')
"""
THe @ operator also allows for multiplying a matrix and vector
"""
x1 = transitionMatrix @ x0
la.showVector([x1], idx='Migration', col=rooms, label='Museum Migration (x1)',
              label2='x1 = Matrix @ Vector = transitionMatrix @ x0')
print(f'Total people: {red}{round(sum(x1), roundVal):,}{rst}\n\n')
"""
We can use this to simulate a random walk for n number of steps
"""
xFinal, xFinalProb = la.simRandomWalk(
    transMatrix=transitionMatrix, x0=x0, col=rooms, steps=100
)


"""
1.4 Eigenvectors:


In the previous section, we simulated a random walk by multiplying by M 100 times. Now 
let’s see what happens if we multiply the result by M one more time.
"""
x = transitionMatrix @ xFinalProb
la.showVector(array=[x], idx='Migration', col=rooms, label='Probability',
              label2='transitionMatrix @ xFinalProb = x', pHeader=True)
"""
Lets compare this vector to xFinalProb
"""
print(f'Comparison: np.allclose(x, xFinalProb) = '
      f'{red}{np.allclose(x, xFinalProb)}{rst}\n'
      f'\tIf {red}True{rst}, then {cyan}Mx = x{rst}\n\n'
      f'\tTherefore {cyan}x{rst} is an {magenta}eigenvector{rst} of {cyan}M{rst}\n'
      f'\t* M is the transitionMatrix\n\n'
      f'\tGenerally, {cyan}x{rst} is an {magenta}eigenvector{rst} of {cyan}M{rst} '
      f'when {cyan}M@x = λ*x{rst}\n'
      f'\t* Where {cyan}λ{rst} is the {magenta}eigenvalue{rst} of {cyan}M{rst}\n\n')
"""
np.allclose(x1, x2):
* True: values are the same within the floating point limit

A "True" result implies that Mx = x, for this particular value of "x"
This means that "x" is an eigenvector of "M"

More generally, "x" is an eigenvector of "M" if Mx = λx, 
    where λ is an eigenvalue of "M".
In this example, the eigenvalue that corresponds to the eigenvector we found is.


The method we used to simulate a random walk, repeated matrix multiplication, is also an 
algorithm for finding an eigenvector.

If you start with almost any vector "x" and repeatedly compute: x = M @ x
the value of "x" converges to an eigenvector of "M" (I’ll explain the “almost” later).

But this method only finds one eigenvector, and in general there can be more than one.
Also it is not particularly efficient.

SciPy provides a function called eig that implements an efficient algorithm that finds 
all of the eigenvectors.
"""
from scipy.linalg import eig
vals, vecs = eig(transitionMatrix)
"""
The result is a one-dimensional array of eigenvalues and a two dimensional array of 
eigenvectors. In this example, there are six eigenvalues.
"""
la.showVector(array=[vals], col=rooms, label='Eigenvalues', pHeader=True)
"""
In general, the eigenvalues of a matrix can be complex numbers, but in this example the 
imaginary parts are all 0, that is, the eigenvalues are all real. 
We can use the abs function to get the real parts.
"""
la.showVector(array=[abs(vals)], col=rooms, label='Eigenvalues', label2='Absolute Values')
"""
The eigenvector we found by repeated multiplication is the one with eigenvalue 1.
We can get the corresponding eigenvector by selecting the first column from vecs
"""
la.showMatrix(array=vecs, idx=rooms, col=rooms, label='Eigenvectors',
              )
vec = vecs[:, selRoomIdx]
la.showVector(array=[vec], vector=selRoom, idx=rooms, col=rooms,
              label=f'Eigenvector', label2=selRoom)
"""
The Lobby vector doesnt look like the vector we found with repeated multiplication, but 
if we divide through so the elements add up to 1, we’ll see that it is.
"""
s = vec.sum()
print(f'Vector sum: {s}\n')
vec = vec / s
la.showVector(array=[vec], idx=selRoom, col=rooms, label=f'Eigenvector',
              label2=f'{selRoom} Normalized')
print(f'Compare {pink}{selRoom} Normalized{rst} to {pink}Simulate Random Walk{rst}:\n'
      f'\tIf {red}True{rst}, then eigenvector!\n'
      f'\tnp.allclose(vec, xFinalProb) = {red}{np.allclose(vec, xFinalProb)}{rst}\n\n')
"""
While we’re at it let’s confirm that the other eigenvectors also work.
For example, here’s the second eigenvector.
"""
# eigenvects = []
# for idx in range(len(rooms)):
#     r = rooms[idx]
#     vecN = vecs[:, idx]
#     valN = vals[idx].real
#     isEV = np.allclose(transitionMatrix@vecN, valN*vecN)
#     eigenvects.append(isEV)
#     if idx != 0:
#         header = True if idx == 1 else False
#         la.showVector(array=[vecN], idx=r, col=rooms, label=f'Additional Eigenvectors',
#                       label2=f'{r}'pHeader=header)
#         print(f'Eigenvalue (valN): {red}{round(valN, 3):,}{rst}\n')
#         print(f'Is {cyan}vec{idx}{rst} and {magenta}eigenvector{rst} of the '
#               f'{magenta}transitionMatrix{rst}:')
#         print(f'\tnp.allclose(transitionMatrix@vecN, valN*vecN) = '
#               f'{red}{isEV}{rst}\n\n')
# la.showVector([eigenvects], idx='Eigenvectors', col=rooms, label=f'Eigenvectors')


"""
1.5 Linearity:


Since there are multiple eigenvectors for this transition matrix, you might wonder why we 
focused on the first one and ignored the rest. The reason is that the eigenvector that 
corresponds to eigenvalue 1 represents a stationary point of M.
That is, a distribution of probability across the nodes that is unchanged when we multiply 
by the transition matrix.

The other eigenvalues are all less than 1, which means that if we start with one of the 
other eigenvectors and multiply by M repeatedly, the result gets smaller and smaller, 
converging toward 0.

Now suppose we start with an initial vector that is the sum of all of the eigenvectors. 
We’ll use this function to add up the columns in an array. In NumPy, axis 0 is down the 
rows and axis 1 is across the columns, so calling sum with axis=1 adds up the columns.
"""
la.showMatrix(array=transitionMatrix, idx=rooms, col=rooms, label='Transition Matrix',
              pHeader=True)
la.sumColumns(array=transitionMatrix, col=rooms, label='Transition Matrix')
la.showMatrix(array=vecs, idx=rooms, col=rooms, label='Eigenvectors')
la.showVector(array=[vals], col=rooms, label='Eigenvalues')
print(f'Transition Matrix: {np.shape(transitionMatrix)}\n'
      f'Eigenvectors:      {np.shape(vecs)}\n'
      f'Eigenvalues:       {np.shape(vals)}\n\n')
print(f'Lets sum the {magenta}Eigenvectors{rst} and then multiply the '
      f'{magenta}Transition Matrix{rst} by the sum {cyan}x0{rst}:')
x0 = la.sumColumns(array=vecs, col=rooms, label='x0 = Eigenvectors.sum(axis=1)')
"""
Noe let's multiply the sum of the Eigenvectors by the Transition Matrix
"""
x1 = transitionMatrix @ x0
la.showVector(array=[x1], idx='x1', col=rooms, label=f'x1 = TransitionMatrix @ x0')
print(f'\nWe can also reach the same result by:\n'
      f'* multiplying the {magenta}Transition Matrix{rst} by the '
      f'{magenta}Eigenvectors{rst}\n'
      f'* then {magenta}summing{rst} the columns\n')
la.sumColumns(array=transitionMatrix@vecs, col=rooms,
              label='x1 = TransitionMatrix @ Eigenvectors')
print(f'The result is a weighted the sum of {magenta}eigenvectors{rst}, where the weights '
      f'are the {magenta}eigenvalues{rst}, which we can compute like this:')
la.sumColumns(array=vals*vecs, col=rooms, label='x1 = vals * vecs', pSpace=True)
"""
Misc:
"""
# la.header()
# print(f'We can also use the {magenta}Eigenvalues{rst} to scale the '
#       f'{magenta}Eigenvectors{rst}.\n'
#       f'This is the "{cyan}λx{rst}" term of the equation:\n\t{cyan}M@x = λ*x{rst}\n')
# la.showMatrix(transitionMatrix@vecs, idx=rooms, col=rooms,
#               label='transitionMatrix @ Eigenvectors')
# """
# Operator: * = element-wise (Hadamard) multiplication.
#     A * B multiplies matching positions independently, no summation.
# """
# vv = vals*vecs
# la.showMatrix(vv.real, idx=rooms, col=rooms, label='Eigenvalues * Eigenvectors',
#               label2='Real Values')


"""
1.6 PageRank:


The algorithm we used to find a friend at the museum is very close to PageRank, the 
algorithm originally used by Google to quantify the importance of web pages. It works by 
making a graph where each node is a web page and each edge represents a link. 
If a page contains a link to another page, the graph contains an edge between the 
corresponding nodes.

In the museum graph, the edges are undirected – that is, if there is a connection from 
room A to room B, there is also a connection from B to A. There are no one-way doors.

In the web graph, the edges are directed – if there is a link from page A to page B, 
there is not necessarily a link from B to A.

As an example, we’ll use a dataset of Wikipedia pages collected as part of a research 
project and made available in the Stanford Large Network Dataset Collection.
For this exercise, you'll need to download the following dataset:
    https://snap.stanford.edu/data/wikispeedia/wikispeedia_paths-and-graph.tar.gz
"""
la.header()
print('PageRank:\n')
path = os.path.join('Data', 'wikispeedia_paths-and-graph.tar.gz')
print(f'Loading file:\n\t{green}{path}{rst}\n')
with tarfile.open(path, 'r:gz') as tar:
    file = tar.getmember('wikispeedia_paths-and-graph/links.tsv')
    f = tar.extractfile(file)
    wikiLinks = pd.read_csv(f, sep='\t', names=['source', 'target'], skiprows=12)
sNum = 5
print(f'Wiki: Sample number {sNum}\n{wikiLinks.sample(sNum)}\n')
"""
We can use the NetworkX function from_pandas_edgelist to make a graph.
The create_using keyword specifies that we want a directed graph, that is, one that keeps 
track of the direction of the edges.
"""
GWiki = nx.from_pandas_edgelist(wikiLinks, create_using=nx.DiGraph)
print(f'N Nodes: {red}{GWiki.number_of_nodes():,}{rst}\n'
      f'N Edges: {red}{GWiki.number_of_edges():,}{rst}')
"""
One of the challenges of working with a directed graph is that it might contain sink 
nodes. For example, if there is a page with no links, it acts like a dead end, if a random 
walk enters a sink node, it can’t leave.

To check for sink nodes, we can use the out_degree function, which computes the number of 
edges that go out from each node.
"""
nodesSink = [unquote(node) for node, degree in GWiki.out_degree() if degree == 0]
print(f'\nSink Nodes: {grey}{", ".join(nodesSink)}{rst}')
print(f'\tA "{grey}Sink Node{rst}" is a webpage with no link, this acts as a dead end.\n'
      f'\tIf a random walk enters the sink, it won\'t be able to leave.\n')
print(f'We\'ll deal with the "Sink Nodes" by occasionally "teleporting" to other nodes.\n'
      f'By mixing up from choosing a random link, and teleporting to other nodes we will '
      f'be adding randomness to the random walk.')
"""
To handle these sink nodes, we’ll add more randomness to the random walk. Instead of 
choosing a random link at every step, we will sometimes “teleport” by choosing another 
node entirely at random.

First, let’s get the adjacency matrix from the graph.
"""
A = nx.to_numpy_array(GWiki)
"""
The row sum of a sink node is 0
"""
rowSums = A.sum(axis=1)
nodesSink = np.where(rowSums == 0)[0]
print(f'\nSink Nodes (N={red}{len(nodesSink):,}{rst}):\n'
      f'{grey}{pd.DataFrame(nodesSink).T}{rst}')
"""
If we get to a sink node, we have to choose a new node at random, so we’ll make a vector 
with an equal probability for every node.
"""
n = len(A)
p = np.repeat(1 / n, n)
print(f'Sink Node probability: {p[0]}')
print(f'\tWe\'ll replace the "Sink Nodes" rows with the teleport probabilities\n'
      f'\tThis allows for a equal probability to jump to any other node.\n')
for node in nodesSink:
    A[node] = p
"""
Now we can compute the transition matrix as we did before.
"""
M = A / A.sum(axis=1, keepdims=True)
"""
At every step of a PageRank random walk, there is some probability of teleporting 
regardless of whether we are in a sink node or not. 

A parameter called alpha specifies the probability of not teleporting.
    The default value is 0.85, which means we teleport 15% of the time.

To make a transition matrix with teleporting, we’ll compute a weighted sum of the 
transition matrix "M" and the teleport vector, "p".
"""
print(f'During a PageRank random walk there is a probability of teleporting regardless '
      f'if we are in a sink or not.\n'
      f'\tThe {cyan}Alpha{rst} param specifies this probability.\n'
      f'\tWhen {cyan}Alpha{rst}={red}0.85{rst}, {cyan}prob_teleport{rst}={red}15%{rst}\n')
alpha = 0.85
GM = alpha * M + (1 - alpha) * p
print(f'GM.shape: {red}{GM.shape}{rst}\n')
"""
The result is called a Google matrix. 
NetworkX provides a function that computes it (with a few more options).
"""
GM_nx = nx.google_matrix(GWiki, alpha=alpha)
print(f'Comparison: np.allclose(GM, GM_nx): {red}{np.allclose(GM, GM_nx)}{rst}\n')

