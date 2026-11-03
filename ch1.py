"""
1. The Power of Linear Algebra

To demonstrate, this chapter introduces vectors, matrices, and matrix-vector
multiplication.

We’ll use these tools to simulate a random walk in a network, and we’ll implement the
PageRank algorithm Google used to revolutionize web search. Along the way, we’ll learn
about eigenvectors, one of the most powerful ideas in linear algebra.
"""

from functions import LinearAlgebra
import numpy as np
import pandas as pd
import sys
from utils import decorate


# Initialize class
la = LinearAlgebra()

"""
1.1. A Day at the Museum

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

"""
Random walk:

Now we will simulate a random walk through the museum. 
We'll use a Counter to keep track of the number of times each room is visited. 
Then divide through by the number of iterations, so the values in the Counter sum to one.
"""
la.randomWalk(G=graph, start='Lobby')


"""
Adjacency Matrix:

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
    matrix=adjacencyMatrix, idx=graph.nodes, col=graph.nodes
)
la.showVector(matrix=transitionMatrix, vector='Lobby', idx=graph.nodes, col=graph.nodes)


"""
Dot Product:

Now suppose there are 100 people in every room. We can create an array that represents 
this hypothetical starting condition.
"""
N = len(graph)
x0 = np.full(N, 100)
print(f'Museum Migration:\n'
      f'{pd.DataFrame([x0], index=['Start Condition'], columns=list(graph.nodes))}')
"""
And now suppose that everyone leaves the room they are in, choosing one of the exits at 
random. To find out how many of them end up in the Lobby, we can multiply row0 by x
"""
index = 0
label = list(graph.nodes)[index]
row0 = transitionMatrix[index]
product = x0 * row0 # Multiply arrays
print(f'\nx0 * {label}:\n'
      f'{pd.DataFrame([product], index=[label], columns=list(graph.nodes))}\n\n'
      f'{round(product.sum(), 1):,} people have moved to the {label}\n\n')

