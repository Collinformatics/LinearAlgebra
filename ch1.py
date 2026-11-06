"""
1. The Power of Linear Algebra

To demonstrate, this chapter introduces vectors, matrices, and matrix-vector
multiplication.

We’ll use these tools to simulate a random walk in a network, and we’ll implement the
PageRank algorithm Google used to revolutionize web search. Along the way, we’ll learn
about eigenvectors, one of the most powerful ideas in linear algebra.
"""

from colorama import Fore, Style
from functions import LinearAlgebra
import numpy as np
import pandas as pd
import sys
from utils import decorate


# Define colors
red = Fore.LIGHTRED_EX
gold = Fore.YELLOW
yellow = Fore.LIGHTYELLOW_EX
cyan = Fore.LIGHTCYAN_EX
blue = Fore.LIGHTBLUE_EX
pink = Fore.LIGHTMAGENTA_EX
white = Fore.LIGHTWHITE_EX
bold = Style.BRIGHT
rst = Style.RESET_ALL

# Set options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:,.3f}'.format)

roundVal = 2

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
    ("Main Stair", "Café"),
    ("Café", "Garden"),
]
graph = la.plotGraph(
    edges=inEdges, title='Museum', figName='ch1_graph', nodeSize=1200, plotFig=inPlotGraph
)
rooms = graph.nodes
"""
Random walk:

Now we will simulate a random walk through the museum. 
We'll use a Counter to keep track of the number of times each room is visited. 
Then divide through by the number of iterations, so the values in the Counter sum to one.
"""
la.randomWalk(G=graph, start='Lobby')


"""
1.2 Adjacency Matrix:

Each element is either 1 to indicate that there is an edge between two nodes or 0 to 
indicate that there is not.
"""
adjacencyMatrix = la.adjacencyMatrix(G=graph)
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
la.showVector(array=transitionMatrix, vector='Lobby', idx=rooms, col=rooms)


"""
1.3 Dot Product:

Now suppose there are 100 people in every room. We can create an array that represents 
this hypothetical starting condition.
"""
N = len(graph)
x0 = np.full(N, 100)
la.showVector(array=[x0], idx='Start Condition', col=rooms,
              label='Museum Migration (x0)')
"""
And now suppose that everyone leaves the room they are in, choosing one of the exits at 
random. To find out how many of them end up in the Lobby, we can multiply row0 by x
"""
index = 0
label = list(rooms)[index]
row0 = transitionMatrix[index]
product = x0 * row0 # Multiply arrays
nMigrants = round(product.sum(), roundVal)
la.showVector(array=[product], idx=[label], col=rooms,
              label=f'x0 * {label}', pSpace=False)
print(f'* A total of {red}{nMigrants:,}{rst} people have moved to the {label}')
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
              label2='x1 = Matrix @ Vector = transitionMatrix @ x0', pSpace=False)
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
la.showVector(array=[x], idx='Migration', col=rooms,
              label='Probability', label2='x = nSteps + 1', pSpace=False)
"""
Lets compare this vector to xFinalProb
"""
print(f'Comparison: np.allclose(x, xFinalProb) = '
      f'{red}{np.allclose(x, xFinalProb)}{rst}\n\n')
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
la.showVector(array=[vals], col=rooms, label='Eigenvalues', pSpace=False)
la.showMatrix(matrix=vecs, idx=rooms, col=rooms, label='Eigenvectors', pHeader=False)
"""
The result is a one-dimensional array of eigenvalues and a two dimensional array of 
eigenvectors. In this example, there are six eigenvalues.


"""

