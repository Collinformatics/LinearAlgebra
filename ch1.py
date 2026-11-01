"""
1. The Power of Linear Algebra

To demonstrate, this chapter introduces vectors, matrices, and matrix-vector
multiplication.

We’ll use these tools to simulate a random walk in a network, and we’ll implement the
PageRank algorithm Google used to revolutionize web search. Along the way, we’ll learn
about eigenvectors, one of the most powerful ideas in linear algebra.
"""

from collections import Counter
from functions import LinearAlgebra
import random
from utils import decorate


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

def randomWalk(G, start, iters=1000):
    print(f'Random Walk:')
    counter = Counter()
    node = start

    for _ in range(iters):
        node = random.choice(list(G[node]))
        counter[node] += 1

    for key in counter:
        counter[key] /= iters

    for k, v in counter.items():
        print(f'* {k}: {v}')
    print()

randomWalk(G=graph, start='Lobby')
