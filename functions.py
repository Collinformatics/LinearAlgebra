from collections import Counter
from colorama import Fore, Style
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import networkx as nx
import numpy as np
import os
import pandas as pd
import random
import subprocess
import sys


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
bold = Style.BRIGHT
rst = Style.RESET_ALL

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
        self.roundVal = 3
        self.rstFloat = False

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

        # Set options
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', 10)
        pd.set_option('display.width', 1000)
        self.formatFloat()


    def formatFloat(self, sciNotation=False):
        if sciNotation:
            pd.options.display.float_format = lambda x: f'{x:,.{self.roundVal}e}'
            self.rstFloat = True
        elif self.rstFloat:
            pd.options.display.float_format = lambda x: f'{x:,.{self.roundVal}f}'
            self.rstFloat = False


    @staticmethod
    def header(n=80):
        print('='*n)


    @staticmethod
    def checkList(d):
        if isinstance(d, nx.classes.reportviews.NodeView):
            d = list(d)
        elif not isinstance(d, list):
            d = [d]
        return d


    def showMatrix(self, matrix, idx='', col='', label='Matrix', label2='',
                   pSpace=True, pHeader=True):
        if pHeader:
            self.header()
        idx = self.checkList(idx)
        col = self.checkList(col)
        if label2:
            print(f'{green}{label}{rst}: {pink}{label2}{rst}')
        else:
            print(f'{green}{label}{rst}:')
        df = pd.DataFrame(matrix, index=idx, columns=col)
        if abs(df.loc[:,:].min(axis=None)) > 10**3:
            self.formatFloat(sciNotation=True)
        print(f'{df}\n')
        self.formatFloat()
        if pSpace:
            print()


    def showVector(self, array, idx='', col='', vector=False, label='Vector',
                   label2='', pSpace=True, pHeader=True):
        if pHeader:
            self.header()
        idx = self.checkList(idx)
        col = self.checkList(col)
        if label2:
            print(f'{gold}{label}{rst}: {pink}{label2}{rst}')
        else:
            print(f'{gold}{label}{rst}:')
        if vector:
            df = pd.DataFrame(array, index=idx, columns=col).loc[[vector], :]
        else:
            df = pd.DataFrame(array, index=idx, columns=col)
        if abs(df.loc[:,:].min(axis=None)) > 10**3:
            self.formatFloat(sciNotation=True)
        print(f'{df}\n')
        self.formatFloat()
        if pSpace:
            print()


    def adjacencyMatrix(self, G):
        array = nx.to_numpy_array(G).astype(int)
        self.showMatrix(matrix=array, idx=G.nodes, col=G.nodes, label='Adjacency Matrix')
        return array


    def transitionMatrix(self, matrix, idx, col):
        m = matrix / matrix.sum(axis=0, keepdims=True)
        self.showMatrix(matrix=m, idx=idx, col=col, label='Transition Matrix')
        return m


    def simRandomWalk(self, transMatrix, x0, steps, col):
        self.header()
        print(f'Simulate Random Walk\n'
              f'N Steps: {pink}{steps:,}{rst}\n')
        x = x0
        for _ in range(steps):
            x = transMatrix @ x # Matrix-vector multiplication
        self.showVector(array=[x], idx='Migration', col=col,
                        label='Population', pSpace=False, pHeader=False)
        n = sum(x)
        print(f'Total people: {red}{round(n, 2):,}{rst}\n')
        prob = x / n
        self.showVector(array=[prob], idx='Migration', col=col,
                        label='Probability', pSpace=False, pHeader=False)
        print()
        return x, prob


    def randomWalk(self, G, start, iterations=1000):
        counter = Counter()
        node = start

        for _ in range(iterations):
            node = random.choice(list(G[node]))
            counter[node] += 1

        for key in counter:
            counter[key] /= iterations

        self.showVector(array=[counter.values()], idx='Probability', col=G.nodes,
                        label='Random Walk')


    def saveFig(self, plot, figureName, fig=None):
        self.header()
        if figureName:
            path = os.path.join(self.figDir, figureName)
            plot.savefig(path, bbox_inches='tight')
            print(f'Figure was saved at path:\n\t{green}{path}{rst}\n')
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
