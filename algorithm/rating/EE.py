from baseclass.IterativeRecommender import IterativeRecommender
import numpy as np
from tool import config


class EE(IterativeRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, fold='[1]'):
        super(EE, self).__init__(conf, trainingSet, testSet, fold)

    # def readConfiguration(self):
    #     super(EE, self).readConfiguration()
    #     Dim = config.LineConfig(self.config['EE'])
    #     self.Dim = int(Dim['-d'])

    def initModel(self):
        super(EE, self).initModel()
        self.Bu = np.random.rand(self.dao.trainingSize()[0])/10  # bias value of user
        self.Bi = np.random.rand(self.dao.trainingSize()[1])/10  # bias value of item
        # self.X = np.random.rand(self.dao.trainingSize()[0], self.Dim)/10
        # self.Y = np.random.rand(self.dao.trainingSize()[1], self.Dim)/10

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                user, item, rating = entry
                error = rating - self.predict(user,item)
                u = self.dao.user[user]
                i = self.dao.item[item]
                self.loss += error ** 2
                self.loss += self.regU * (self.P[u] - self.Q[i]).dot(self.P[u] - self.Q[i])
                bu = self.Bu[u]
                bi = self.Bi[i]
                #self.loss += self.regB * bu ** 2 + self.regB * bi ** 2
                # update latent vectors
                self.P[u] -= self.lRate * (error + self.regU) * (self.P[u] - self.Q[i])
                self.Q[i] += self.lRate * (error + self.regI) * (self.P[u] - self.Q[i])
                self.Bu[u] += self.lRate * (error - self.regB * bu)
                self.Bi[i] += self.lRate * (error - self.regB * bi)
            self.loss+=self.regB*(self.Bu*self.Bu).sum()+self.regB*(self.Bi*self.Bi).sum()
            iteration += 1
            self.isConverged(iteration)

    def predict(self, u, i):
        if self.dao.containsUser(u) and self.dao.containsItem(i):
            u = self.dao.user[u]
            i = self.dao.item[i]
            return self.dao.globalMean + self.Bi[i] + self.Bu[u] - (self.P[u] - self.Q[i]).dot(self.P[u] - self.Q[i])
        else:
            return self.dao.globalMean

    def predictForRanking(self,u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            u = self.dao.user[u]
            res = ((self.Q-self.P[u])*(self.Q-self.P[u])).sum(axis=1)+self.Bi+self.Bu[u]+self.dao.globalMean
            return res
        else:
            return [self.dao.globalMean]*len(self.dao.item)

