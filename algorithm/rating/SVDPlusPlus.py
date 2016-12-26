from baseclass.IterativeRecommender import IterativeRecommender
import numpy as np
import math
class SVDPlusPlus(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(SVDPlusPlus, self).__init__(conf,trainingSet,testSet,fold)

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
                u, i, r = entry
                itemIndexs, rating = self.dao.userRated(u)
                w = len(itemIndexs)
                error = r - self.predict(u, i)
                u = self.dao.getUserId(u)
                i = self.dao.getItemId(i)
                self.loss += error ** 2
                p = self.P[u].copy()
                q = self.Q[i].copy()
                self.loss += self.regU * p.dot(p) + self.regI * q.dot(q)
                bu = self.Bu[u]
                bi = self.Bi[i]
                self.loss += self.regB * bu ** 2 + self.regB * bi ** 2
                #update latent vectors
                self.Bu[u] += self.lRate*(error-self.regB*bu)
                self.Bi[i] += self.lRate*(error-self.regB*bi)
                for j in itemIndexs:
                    y = self.Y[j].copy()
                    # self.loss += self.regU * y.dot(y)
                    try:
                        self.Q[i] += self.lRate * error * self.Y[j]/w
                        #self.Y[j] += self.lRate * (error * q / w - self.regI * q)
                    except ZeroDivisionError:
                        pass
                self.P[u] += self.lRate * (error * q - self.regU * p)
                self.Q[i] += self.lRate * (error * p - self.regI * q)
            self.loss *= 0.5
            iteration += 1
            self.isConverged(iteration)

    def predict(self,u,i):
        pred = 0
        if self.dao.containsUser(u) and self.dao.containsItem(i):
            itemIndexs,rating = self.dao.userRated(u)
            w = len(itemIndexs)
            # w = math.sqrt(len(itemIndexs))
            u = self.dao.getUserId(u)
            i = self.dao.getItemId(i)
            for j in itemIndexs:
                try:
                    pred += (self.Y[j].dot(self.Q[i]))/w
                except ZeroDivisionError:
                    pass
            pred += self.P[u].dot(self.Q[i]) + self.dao.globalMean + self.Bi[i] + self.Bu[u]

        else:
            pred = self.dao.globalMean
        return pred
