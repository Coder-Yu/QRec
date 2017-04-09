from baseclass.IterativeRecommender import IterativeRecommender
import numpy as np
from tool import config
from collections import defaultdict
from math import log,exp

class CoFactor(IterativeRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, fold='[1]'):
        super(CoFactor, self).__init__(conf, trainingSet, testSet, fold)

    def readConfiguration(self):
        super(CoFactor, self).readConfiguration()
        extraSettings = config.LineConfig(self.config['CoFactor'])
        self.negCount = int(extraSettings['-k']) #the number of negative samples
        if self.negCount < 1:
            self.negCount = 1
        self.regR = float(extraSettings['-gamma'])
        self.filter = int(extraSettings['-filter'])

    def printAlgorConfig(self):
        super(CoFactor, self).printAlgorConfig()
        print 'Specified Arguments of', self.config['recommender'] + ':'
        print 'k: %d' % self.negCount
        print 'regR: %.5f' %self.regR
        print 'filter: %d' %self.filter
        print '=' * 80

    def initModel(self):
        super(CoFactor, self).initModel()
        self.w = np.random.rand(self.dao.trainingSize()[1])/10  # bias value of item
        self.c = np.random.rand(self.dao.trainingSize()[1])/10  # bias value of context

        self.G = np.random.rand(self.dao.trainingSize()[1], self.k)/10 #context embedding

        #constructing SPPMI matrix
        self.SPPMI = defaultdict(dict)
        D = len(self.dao.item)
        print 'Constructing SPPMI matrix...'
        #for larger data set has many items, the process will be time consuming
        maxVal = 0
        for item1 in self.dao.item:
            uList1, rList1 = self.dao.itemRated(item1)
            if len(uList1)<self.filter:
                continue
            for item2 in self.dao.item:
                if item1==item2:
                    continue
                if not self.SPPMI[item1].has_key(item2):
                    uList2,rList2 = self.dao.itemRated(item2)
                    if len(uList2)<self.filter:
                        continue
                    count = len(set(uList1).intersection(set(uList2)))
                    val = 0
                    if count>0:
                        val = max([log(float(count*D)/(len(uList1)*len(uList2)),2)-log(self.negCount,2),0])
                    if val > 0:
                        if maxVal < val:
                            maxVal = val
                        self.SPPMI[item1][item2] = val
                        self.SPPMI[item2][item1] = self.SPPMI[item1][item2]

        #normalize
        for item1 in self.SPPMI:
            for item2 in self.SPPMI[item1]:
                self.SPPMI[item1][item2] = self.SPPMI[item1][item2]/maxVal


    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                user, item, rating = entry
                error = rating - self.predict(user,item)
                u = self.dao.getUserId(user)
                i = self.dao.getItemId(item)
                p = self.P[u].copy()
                q = self.Q[i].copy()
                self.loss += error ** 2 + self.regU*p.dot(p) + self.regI*q.dot(q)
                #update latent vectors
                self.P[u] += self.lRate * (error * q - self.regU * p)
                self.Q[i] += self.lRate * (error * p - self.regI * q)
            for item in self.SPPMI:
                i = self.dao.getItemId(item)
                for context in self.SPPMI[item]:
                    j = self.dao.getItemId(context)
                    m = self.SPPMI[item][context]
                    g = self.G[j]
                    diff = (m - q.dot(g) - self.w[i] - self.c[j])
                    self.loss += diff ** 2 + self.regR * g.dot(g)
                    # update latent vectors
                    self.Q[i] += self.lRate * diff * g
                    self.G[j] += self.lRate * diff * q
                    self.w[i] += self.lRate * diff
                    self.c[j] += self.lRate * diff
            iteration += 1
            self.isConverged(iteration)


