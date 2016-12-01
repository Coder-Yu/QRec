from baseclass.IterativeRecommender import IterativeRecommender
import numpy as np

class SVD(IterativeRecommender):
    def __init__(self,conf):
        super(SVD, self).__init__(conf)

    def initModel(self):
        super(SVD, self).initModel()
        self.Bu = np.random.rand(self.dao.trainingSize()[0])  # biased value of user
        self.Bi = np.random.rand(self.dao.trainingSize()[1])  # biased value of item

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                u, i, r = entry
                u = self.dao.getUserId(u)
                i = self.dao.getItemId(i)
                error = r-self.P[u].dot(self.Q[i])-self.dao.globalMean-self.Bi[i]-self.Bu[u]
                self.loss+=error**2
                p = self.P[u].copy()
                q = self.Q[i].copy()
                self.loss += self.regU * p.dot(p) + self.regI * q.dot(q)
                bu = self.Bu[u]
                bi = self.Bi[i]
                self.loss += self.regB*bu**2 + self.regB*bi**2
                #update latent vectors
                self.P[u] += self.lRate*(error*q-self.regU*p)
                self.Q[i] += self.lRate*(error*p-self.regI*q)
                self.Bu[u] += self.lRate*(error-self.regB*bu)
                self.Bi[i] += self.lRate*(error-self.regB*bi)
            iteration += 1
            self.isConverged(iteration)

    def predict(self,u,i):
        if self.dao.containsUser(u) and self.dao.containsItem(i):
            u = self.dao.getUserId(u)
            i = self.dao.getItemId(i)
            return self.P[u].dot(self.Q[i])+self.dao.globalMean+self.Bi[i]+self.Bu[u]
        else:
            return self.dao.globalMean
