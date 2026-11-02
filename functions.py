from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import networkx as nx
import numpy as np
import os
import pandas as pd
import random
import subprocess
import sys


# Set options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:,.3f}'.format)

# GPU Usage
# print(f'Jax Device: {jax.default_backend()}')


def pressKey(event):
    if event.key == 'escape':
        plt.close()
    elif event.key == 'e':
        sys.exit()
    elif event.key == 'r':
        python = sys.executable
        os.execl(python, python, *sys.argv)


def getAxisTicks(values, largeValues=True, customSteps=None):
    if customSteps is None:
        customSteps = []
    padding = 0.05
    if not isinstance(values, list):
        values = list(values)
    if customSteps is not None:
        steps = customSteps
    else:
        if largeValues:
            steps = [25, 20, 30, 40, 50]
        else:
            steps = [5, 2, 3, 4]
            padding = 0.01

    dist = values[-1] - values[0]
    padding = int(padding * dist)
    ticks = []
    for step in steps:
        if dist % step == 0:
            ticks = list(np.arange(values[0], values[-1] + step, step))
            print(f'Step: {step}')
            break
    # print(f'xTicks: {ticks}\n')
    return ticks, padding


class LinearAlgebra:
    def __init__(self):
        self.printN = 10

        # Params: Figures
        self.figDir = 'Figures'
        self.figSize = (9.5, 8) # (width, height)
        self.figSizeWide = (9.5, 5)
        self.labelSizeTitle = 20 # Set font-sizeself.figDir='Figures'
        self.labelSizeAxis = 16 # Set font-size
        self.labelSizeTicks = 12 # Set font-size
        self.lineThickness = 1.5
        self.tickLength = 4
        self.figureResolution = 600

        if not os.path.exists(self.figDir):
            os.makedirs(self.figDir, exist_ok=True)


    @staticmethod
    def adjacencyMatrix(G):
        print(f'Adjacency Matrix:')
        array = nx.to_numpy_array(G).astype(int)
        print(f'Array:\n{array}\n')
        df = pd.DataFrame(array, index=G.nodes, columns=G.nodes)
        print(f'{df}\n\n')


    @staticmethod
    def randomWalk(G, start, iters=1000):
        print(f'Random Walk:')
        counter = Counter()
        node = start

        for _ in range(iters):
            node = random.choice(list(G[node]))
            counter[node] += 1

        for key in counter:
            counter[key] /= iters

        d = pd.DataFrame(counter.values(), index=list(counter.keys()),
                         columns=['Probability'])
        print(f'{d}\n\n')


    def saveFig(self, plot, figureName, fig=None):
        if figureName:
            path = os.path.join(self.figDir, figureName)
            plot.savefig(path, bbox_inches='tight')
            print(f'Figure was saved at path:\n\t{path}\n')
            subprocess.run(['xdg-open', path])  # Open fig in web browser
        else:
            fig.canvas.mpl_connect('key_press_event', pressKey)
            plt.show()


    def plotGraph(self, edges, title, figName, nodeSize = 1000, plotFig=True):
        G = nx.Graph()
        G.add_edges_from(edges)
        fig, ax = plt.subplots(figsize=self.figSize)
        nx.draw(
            G, ax=ax, with_labels=True,
            node_size=nodeSize, node_color='lightgreen',
            font_size=14, font_color='black', font_weight='bold',
            width=2.0, edge_color='black'
        )
        ax.set_title(title, fontsize=self.labelSizeTitle, fontweight='bold')
        plt.tight_layout()

        # Save figure
        if plotFig:
            if '.' not in figName:
                figName += '.png'
            self.saveFig(plot=plt, figureName=figName)

        return G


    def plotLines(self, data, colors, title,
                  labelX, labelY, yRotate=0, browserFig=False):
        fig, ax = plt.subplots(figsize=self.figSize)
        for i in range(data.columns.size):
            ax.plot(data.index, data.iloc[:,i], color=colors[i],
                    label=data.columns[i], linewidth=self.lineThickness)
        ax.legend(fontsize=self.labelSizeTicks, loc='best',
                  edgecolor='black', framealpha=0.8)

        # Styling
        ax.set_title(title, fontsize=self.labelSizeTitle, fontweight='bold')
        ax.set_xlabel(labelX, fontsize=self.labelSizeAxis)
        ax.set_ylabel(labelY, fontsize=self.labelSizeAxis, labelpad=20, rotation=yRotate)
        ax.tick_params(labelsize=12)

        # Axis params
        xTicks, pad = getAxisTicks(values=data.index)
        if xTicks:
            ax.set_xlim(data.index[0]-pad, data.index[-1]+pad)
            ax.set_xticks(xTicks)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if max(ax.get_ylim()) > 1:
            ax.set_ylim(top=int(np.ceil(ax.get_ylim()[1])))

        # Grid
        ax.grid(True, linewidth=0.25, color='black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for _, spine in ax.spines.items():
            spine.set_visible(True)

        plt.tight_layout()
        self.saveFig(plot=plt, figureName=browserFig, fig=fig)
