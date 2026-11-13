from collections import Counter
from colorama import Fore, Style
import math
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
    def __init__(self, overwriteFigs=False):
        self.printN = 10
        self.roundVal = 3
        self.headerLen = 80
        self.rstFloat = True
        self.webBrowserFigs = True
        self.overwriteFigs = overwriteFigs

        # Params: Figures
        self.figDir = 'Figures'
        self.figSize = (9.5, 8) # (width, height)
        self.figSizeWide = (9.5, 5)
        self.labelSizeTitle = 20  # Set font-size
        self.labelSizeAxis = 16  # Set font-size
        self.labelSizeTicks = 14  # Set font-size
        self.labelSizeLegend = 13
        self.lineThickness = 1.5
        self.tickLength = 4
        self.figureResolution = 600
        
        self.mkdir(self.figDir)

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
    def formatList(d):
        if isinstance(d, nx.classes.reportviews.NodeView):
            d = list(d)
        elif not isinstance(d, list):
            d = [d]
        return d


    @staticmethod
    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)


    def header(self, txt=''):
        if txt:
            l = len(txt) + 2
            b = int(np.floor((self.headerLen - l) / 2))
            if l % 2 == 0:
                print(f'{"=" * b} {txt} {"=" * b}')
            else:
                print(f'{"=" * b} {txt} {"=" * (b + 1)}')
        else:
            print('=' * self.headerLen)


    @staticmethod
    def roundUp(v, m=5):
        """
        :param v: round this value
        :param m: round value to a multiple of m
        """
        return int(np.ceil(v / m) * m)


    def showMatrix(self, array, idx='', col='', label='Matrix', label2='',
                   pHeader=False, pSpace=False, sumColumns=False):
        if pHeader:
            self.header()
        # if not idx and len(array) > 1:
        #     idx = [i for i in range(len(array))]
        # else:
        idx = self.formatList(idx)
        col = self.formatList(col)
        if label2:
            print(f'{green}{label}{rst}: {pink}{label2}{rst}')
        else:
            print(f'{green}{label}{rst}:')
        df = pd.DataFrame(array, index=idx, columns=col)
        if abs(df.loc[:,:].min(axis=None)) > 10**3:
            self.formatFloat(sciNotation=True)
        if sumColumns:
            df.loc['** Sum **', :] = list(df.sum(axis=1))
        print(f'{df}\n')
        self.formatFloat()
        if pSpace:
            print()


    def showVector(self, array, idx='', col='', vector=False, label='Vector',
                   label2='', pHeader=False, pSpace=False):
        if pHeader:
            self.header()
        idx = self.formatList(idx)
        col = self.formatList(col)
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


    def sumColumns(self, array, col, label='', pData=True, pHeader=False,
                   pSpace=False):
        s = list(array.sum(axis=1))
        if pData:
            self.showVector(array=[s], idx='Sum', col=col, label='Sum Columns',
                            label2=label, pHeader=pHeader, pSpace=pSpace)
        return s


    def adjacencyMatrix(self, G, pHeader=False):
        array = nx.to_numpy_array(G).astype(int)
        self.showMatrix(array=array, idx=G.nodes, col=G.nodes, label='Adjacency Matrix',
                        pHeader=pHeader)
        return array


    def transitionMatrix(self, matrix, idx, col, pHeader=False, pSpace=False):
        m = matrix / matrix.sum(axis=0, keepdims=True)
        self.showMatrix(array=m, idx=idx, col=col, label='Transition Matrix',
                        pHeader=pHeader, pSpace=pSpace)
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


    def randomWalk(self, G, start, iterations=1000, pHeader=False,
                   pSpace=False):
        if pHeader:
            self.header()
        counter = Counter()
        node = start

        for _ in range(iterations):
            node = random.choice(list(G[node]))
            counter[node] += 1

        for key in counter:
            counter[key] /= iterations

        self.showVector(array=[counter.values()], idx='Probability', col=G.nodes,
                        label='Random Walk', pSpace=pSpace)


    def getFigPath(self, figureName, figDir=''):
        if figDir:
            d = os.path.join(self.figDir, figDir)
            path = os.path.join(d, figureName)
            self.mkdir(d)
        else:
            path = os.path.join(self.figDir, figureName)
        if not path.endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.splitext(path)[0] + '.png'
        return path


    def saveFig(self, plot, figureName, pathDir):
        path = self.getFigPath(figureName, pathDir)
        if os.path.exists(path) and not self.overwriteFigs:
            print(f'{yellow}Figure was already saved at path, the figure was not '
                  f'overwritten:\n\t{path}{rst}\n')
        else:
            plot.savefig(path, bbox_inches='tight')
            print(f'Figure was saved at path:\n\t{green}{path}{rst}\n')

    def plotFig(self, plot, figureName, pathDir='', fig=None):
        if self.webBrowserFigs:
            path = self.getFigPath(figureName, pathDir)
            if os.path.exists(path):
                plot.savefig(path, bbox_inches='tight')
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


    def getAxisTicks(self, values, largeValues=True, customSteps=None,
                     addZero=True):
        values, roundBy = self.formatList(values), 5
        maxVal, padding, padAdj = int(max(values)), 0.5, False
        if customSteps is None:
            customSteps = []
        if customSteps:
            steps = customSteps
        else:
            if largeValues:
                steps = [25, 20, 30, 40, 50, 25, 15, 45, 75, 10, 5]
            else:
                steps = [4, 3, 2, 1]
        dist, ticks = self.roundUp(maxVal - values[0], m=roundBy), []
        for step in steps:
            if dist % step == 0:
                ticks = list(np.arange(values[0], maxVal + step, step))
                if len(ticks) > 3:
                    padAdj = step
                    break
        if ticks and addZero and 0 not in ticks:
            ticks = np.insert(ticks, np.searchsorted(ticks, 0), 0)
        if max(ticks) < np.ceil(max(values)) - 1:
            self.header()
            print(f'\nNEW: max ({max(ticks)}) < np.ceil(max(values))-1'
                  f' ({np.ceil(max(values))-1})\n\n'
                  f'Ticks: {ticks}')
            space = ticks[-1] - ticks[-2]
            ticks = np.insert(ticks, np.searchsorted(ticks, max(ticks) + space), 0)
            print(f'\nTicks: {ticks}\n')
            self.header()
            print()
        if padAdj:
            padding = padAdj / 10

        return list(ticks), padding


    def plotLines(self, data, colors, title, figName, labelX, labelY, yRotate=0,
                  figDir='', addGrid=True, saveFig=True):
        """
        :param data: [pd.DataFrame or list of pd.DataFrames]
        :param title: [str] figure title
        :param colors: [list] colors for the lines
        :param figName: [str] file name for the saved figure
        :param labelX: [str] axis label
        :param labelY: [str] axis label
        :param yRotate: [int] rotate the y-label
        :param figDir: [str] save the fig at self.figDir/{figDir}/
            if an empty string is provided the fig will be saved at self.figDir/
        :param addGrid: [bool] add grid to the figure
        :param saveFig: [bool] True: saves the fig, False: does not save the fig
        """

        # Verify that there are a sufficient number of strings
        colors = self.formatList(colors)
        df = True
        if isinstance(data, pd.DataFrame):
            numCurves = len(data.columns)
        else:
            numCurves = 0
            for d in data:
                numCurves += len(d.columns)
            df = False
        c = ['black', 'crimson', 'gold', 'darkgreen', 'lightgreen', 'limegreen',
             'teal', 'turquoise', 'royalblue', 'indigo', 'darkviolet', 'gray']
        diff, n = numCurves - len(colors), len(c)
        if diff > n:
            for _ in range(diff):
                colors.append(c[random.randint(0, n - 1)])
        else:
            i = 0
            while diff + 1 > len(colors):
                if i > 30:
                    break
                clr = c[random.randint(0, n - 1)]
                if clr not in colors:
                    colors.append(clr)
                i += 1
            diff = numCurves - len(colors)
            if diff != 0:
                for _ in range(diff):
                    colors.append(c[random.randint(0, n - 1)])

        # Plot figure
        fig, ax = plt.subplots(figsize=self.figSize)
        xTicks = None
        if df:
            xTicks = pd.Series(data.index)
            for i in range(numCurves):
                ax.plot(data.index, data.iloc[:, i], color=colors[i],
                        label=data.columns[i], linewidth=self.lineThickness)
        else:
            i = 0
            for idx, d in enumerate(data):
                if xTicks is None:
                    xTicks = pd.Series(d.index)
                for idxC in range(len(d.columns)):
                    ax.plot(d.index, d.iloc[:, idxC], color=colors[i],
                            label=d.columns[idxC], linewidth=self.lineThickness)
                    i += 1
        ax.legend(fontsize=self.labelSizeLegend, loc='best', edgecolor='black',
                  framealpha=0.8)
        ax.axhline(y=0, color='black', linewidth=self.lineThickness)

        # Axis params
        ax.set_title(title, fontsize=self.labelSizeTitle, fontweight='bold')
        ax.set_xlabel(labelX, fontsize=self.labelSizeAxis)
        ax.set_ylabel(labelY, fontsize=self.labelSizeAxis, labelpad=20, rotation=yRotate)
        ax.tick_params(labelsize=self.labelSizeTicks, width=self.lineThickness)

        # Set: x-ticks
        large = True if max(xTicks) >= 10 else False
        xTicks, pad = self.getAxisTicks(values=xTicks, largeValues=large)
        if 0 not in xTicks:
            xTicks = np.insert(xTicks, np.searchsorted(xTicks, 0), 0)
        ax.set_xlim(xTicks[0] - pad, xTicks[-1] + pad)
        ax.set_xticks(xTicks)
        if len(xTicks) > 12:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # print(f'X-Ticks: {ax.get_xticks()}\n'
        #       f'X-Lim: {ax.get_xlim()}\n')

        # Set: y-ticks
        yTicks = ax.get_yticks()
        if len(yTicks) > 0:
            large = True if max(yTicks) >= 10 else False
            yTicks, _ = self.getAxisTicks(values=yTicks, largeValues=large)
            if len(yTicks) > 0:
                ax.set_ylim(yTicks[0], yTicks[-1])
                ax.set_yticks(yTicks)
        if len(yTicks) < 1 or len(yTicks) > 12:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Grid
        if addGrid:
            ax.grid(True, linewidth=self.lineThickness, color='lightgrey')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(self.lineThickness)

        plt.tight_layout()
        if saveFig:
            self.saveFig(plot=plt, figureName=figName, pathDir=figDir)
        self.plotFig(
            plot=plt, figureName=figName, pathDir=figDir, fig=fig
        )
