from base.iterativeRecommender import IterativeRecommender
import numpy as np
from util import config
import math
class SVDPlusPlus(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(SVDPlusPlus, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(SVDPlusPlus, self).readConfiguration()
        regY = config.OptionConf(self.config['SVDPlusPlus'])
        self.regY = float( regY['-y'])

    def printAlgorConfig(self):
        super(SVDPlusPlus, self).printAlgorConfig()
        print('Specified Arguments of', self.config['model.name'] + ':')
        print('regY: %.3f' % self.regY)
        print('=' * 80)

    def initModel(self):
        super(SVDPlusPlus, self).initModel()
        self.Bu = np.random.rand(self.data.trainingSize()[0])  # biased value of user
        self.Bi = np.random.rand(self.data.trainingSize()[1])  # biased value of item
        self.Y = np.random.rand(self.data.trainingSize()[1], self.emb_size)

    def trainModel(self):
        epoch = 0
        while epoch < self.maxEpoch:
            self.loss = 0
            for entry in self.data.trainingData:
                user, item, rating = entry
                items, ratings = self.data.userRated(user)
                w = len(items)
                #w = math.sqrt(len(itemIndexs))
                error = rating - self.predictForRating(user, item)
                u = self.data.user[user]
                i = self.data.item[item]
                self.loss += error ** 2
                p = self.P[u]
                q = self.Q[i]
                bu = self.Bu[u]
                bi = self.Bi[i]

                #update latent vectors
                self.Bu[u] += self.lRate*(error-self.regB*bu)
                self.Bi[i] += self.lRate*(error-self.regB*bi)
                sum = 0
                if w> 1:
                    indexes = []
                    for j in items:
                        j = self.data.item[j]
                        if i!=j:
                            indexes.append(j)

                    y = self.Y[indexes]
                    sum += y.sum(axis=0)
                    self.Y[indexes] += self.lRate * (error * q / (w-1) - self.regY * y)
                    self.Q[i] += self.lRate * error * sum/(w-1)

                self.P[u] += self.lRate * (error * q - self.regU * p)
                self.Q[i] += self.lRate * (error * p - self.regI * q)


            self.loss+=self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum() \
               + self.regY*(self.Y*self.Y).sum() + self.regB*((self.Bu*self.Bu).sum()+(self.Bi*self.Bi).sum())
            epoch += 1
            self.isConverged(epoch)


    def predictForRating(self, u, i):
        pred = 0
        if self.data.containsUser(u) and self.data.containsItem(i):
            itemIndexs,rating = self.data.userRated(u)
            w = len(itemIndexs)
            # w = math.sqrt(len(itemIndexs))
            u = self.data.user[u]
            i = self.data.item[i]
            sum = 0
            if w> 0:
                for j in itemIndexs:
                    j = self.data.item[j]
                    sum += self.Y[j]
                pred+= (sum/w).dot(self.Q[i])
            pred += self.P[u].dot(self.Q[i]) + self.data.globalMean + self.Bi[i] + self.Bu[u]

        else:
            pred = self.data.globalMean
        return pred

    def predictForRanking(self,u):
        pred = 0
        if self.data.containsUser(u):
            itemIndexs, rating = self.data.userRated(u)
            w = len(itemIndexs)
            # w = math.sqrt(len(itemIndexs))
            u = self.data.user[u]
            sum = 0
            if w > 0:
                for j in itemIndexs:
                    j = self.data.item[j]
                    sum += self.Y[j]
                pred += self.Q.dot(sum / w)
            pred += self.Q.dot(self.P[u]) + self.data.globalMean + self.Bi + self.Bu[u]

        else:
            pred = [self.data.globalMean] * len(self.data.item)
        return pred
