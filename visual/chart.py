import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import os.path

class Chart(object):
    def __init__(self):
        pass

    @staticmethod
    def distribution(y, title='',xLabel='',yLabel=''):
        f, ax1 = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
        sns.set(style="white")
        x = np.linspace(0, len(y), len(y))
        y.sort(reverse = True)
        plt.plot(x, y, color='green')
        ax1.set_xlabel(xLabel, fontsize=18)
        ax1.set_ylabel(yLabel, fontsize=18)
        ax1.set_xlim(0, len(y))
        # ax1.set_ylim(0,25000)
        ax1.set_title(title, fontsize=22)
        plt.grid(True)
        plt.show()
        plt.close()

    @staticmethod
    def scatter(x, y, color, title='',xLabel='',yLabel=''):
        f, ax1 = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
        # sns.set(style="white")
        ax1.set_ylim(-10, max(y) + 20)
        ax1.set_xlim(-10, max(x) + 20)
        ax1.set_title(title, fontsize=22)
        ax1.set_xlabel(xLabel, fontsize=18)
        ax1.set_ylabel(yLabel, fontsize=18)
        ax1.tick_params(axis='x', labelsize=18)
        ax1.tick_params(axis='y', labelsize=18)
        plt.scatter(x, y, c=color, alpha=0.7, )
        plt.grid(True)
        plt.show()
        plt.close()

    @staticmethod
    def hist(x,y, bins, color,title='',xLabel='',yLabel=''):
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
        sns.set(style="white")
        ax1.grid(True)
        ax1.set_title(title, fontsize=22)
        ax1.set_xlabel(xLabel, fontsize=18)
        ax1.set_ylabel(yLabel, fontsize=18)
        ax1.tick_params(axis='x', labelsize=18)
        ax1.tick_params(axis='y', labelsize=18)
        ind = np.arange(0,1,1.0/len(x))
        ax1.set_xticks(ind+1.0/(2*len(x)))
        ax1.set_xticklabels(x)
        ax1.hist(y, bins, color=color)
        plt.show()
        plt.close()
