from baseclass.IterativeRecommender import IterativeRecommender
import numpy as np
from tool import config
import math
class SVDPlusPlus(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(SVDPlusPlus, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(SVDPlusPlus, self).readConfiguration()
        regY = config.LineConfig(self.config['SVDPlusPlus'])
        self.regY = float( regY['-y'])


    def printAlgorConfig(self):
        super(SVDPlusPlus, self).printAlgorConfig()
        print 'Specified Arguments of', self.config['recommender'] + ':'
        print 'regY: %.3f' % self.regY
        print '=' * 80

    def initModel(self):
        super(SVDPlusPlus, self).initModel()
        self.Bu = np.random.rand(self.dao.trainingSize()[0])  # biased value of user
        self.Bi = np.random.rand(self.dao.trainingSize()[1])  # biased value of item
        self.Y = np.random.rand(self.dao.trainingSize()[1], self.k)


    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                user, item, rating = entry
                itemIndexs, ratings = self.dao.userRated(user)
                w = len(itemIndexs)
                #w = math.sqrt(len(itemIndexs))
                error = rating - self.predict(user, item)
                u = self.dao.user[user]
                i = self.dao.item[item]
                self.loss += error ** 2
                p = self.P[u]
                q = self.Q[i]
                self.loss += self.regU * p.dot(p) + self.regI * q.dot(q)
                bu = self.Bu[u]
                bi = self.Bi[i]
                self.loss += self.regB * bu ** 2 + self.regB * bi ** 2
                #update latent vectors
                self.Bu[u] += self.lRate*(error-self.regB*bu)
                self.Bi[i] += self.lRate*(error-self.regB*bi)
                sum = 0
                if w> 0:
                    for j in itemIndexs:
                        y = self.Y[j].copy()
                        self.loss += self.regY * y.dot(y)
                        sum += y
                        self.Y[j] += self.lRate * (error * q / w - self.regY * y)
                    self.Q[i] += self.lRate * error * sum/w

                self.P[u] += self.lRate * (error * q - self.regU * p)
                self.Q[i] += self.lRate * (error * p - self.regI * q)
            iteration += 1
            self.isConverged(iteration)

    def predict(self,u,i):
        pred = 0
        if self.dao.containsUser(u) and self.dao.containsItem(i):
            itemIndexs,rating = self.dao.userRated(u)
            w = len(itemIndexs)
            # w = math.sqrt(len(itemIndexs))
            u = self.dao.user[u]
            i = self.dao.item[i]
            sum = 0
            if w> 0:
                for j in itemIndexs:
                    sum += self.Y[j]
                pred+= (sum/w).dot(self.Q[i])
            pred += self.P[u].dot(self.Q[i]) + self.dao.globalMean + self.Bi[i] + self.Bu[u]

        else:
            pred = self.dao.globalMean
        return pred

    def predictForRanking(self,u):
        pred = 0
        if self.dao.containsUser(u):
            itemIndexs, rating = self.dao.userRated(u)
            w = len(itemIndexs)
            # w = math.sqrt(len(itemIndexs))
            u = self.dao.user[u]
            sum = 0
            if w > 0:
                for j in itemIndexs:
                    sum += self.Y[j]
                pred += self.Q.dot(sum / w)
            pred += self.Q(self.P[u]) + self.dao.globalMean + self.Bi + self.Bu[u]

        else:
            pred = np.array([self.dao.globalMean] * len(self.dao.item))
        return pred