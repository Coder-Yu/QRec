import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
class Chart(object):
    def __init__(self):
        pass

    @staticmethod
    def distribution(y, title='',xLabel='',yLabel='',savePath='../visual/visualization/p1'):
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6), sharex=True)

        #sns.set(style="white")
        x = np.linspace(0, len(y), len(y))
        y.sort(reverse = True)
        plt.plot(x, y, color='green')
        ax1.set_xlabel(xLabel, fontsize=16)
        ax1.set_ylabel(yLabel, fontsize=16)
        ax1.set_xlim(0, len(y))
        # ax1.set_ylim(0,25000)
        ax1.set_title(title, fontsize=20)
        ax1.tick_params(axis='x', labelsize=16)
        ax1.tick_params(axis='y', labelsize=16)
        plt.grid(True)
        plt.savefig(savePath,bbox_inches='tight')
        #plt.show()
        plt.close('all')

    @staticmethod
    def scatter(x, y, color, title='',xLabel='',yLabel='',savePath='../visual/visualization/p2'):
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6), sharex=True)

        # sns.set(style="white")
        ax1.set_ylim(-10, max(y) + 20)
        ax1.set_xlim(-10, max(x) + 20)
        ax1.set_title(title, fontsize=20)
        ax1.set_xlabel(xLabel, fontsize=16)
        ax1.set_ylabel(yLabel, fontsize=16)
        ax1.tick_params(axis='x', labelsize=16)
        ax1.tick_params(axis='y', labelsize=16)
        plt.scatter(x, y, c=color, alpha=0.7, )
        plt.grid(True)
        plt.savefig(savePath,bbox_inches='tight')
        #plt.show()
        plt.close('all')

    @staticmethod
    def hist(x,y, bins, color,title='',xLabel='',yLabel='',savePath='../visual/visualization/p3'):
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        #sns.set(style="white")

        ax1.grid(True)
        ax1.set_title(title, fontsize=20)
        ax1.set_xlabel(xLabel, fontsize=16)
        ax1.set_ylabel(yLabel, fontsize=16)
        ax1.tick_params(axis='x', labelsize=16)
        ax1.tick_params(axis='y', labelsize=16)
        ind = np.arange(0,1,1.0/len(x))
        ax1.set_xticks(ind+1.0/(2*len(x)))
        ax1.set_xticklabels(x)
        ax1.hist(y, bins, color=color)
        plt.savefig(savePath,bbox_inches='tight')
        #plt.show()
        plt.close('all')
