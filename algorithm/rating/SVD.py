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
            for triple in self.dao.triple:
                u,i,r = triple
                u = self.dao.user[u]
                i = self.dao.item[i]
                error = r-self.P[u].dot(self.Q[i])-self.dao.globalMean-self.Bi[i]-self.Bu[u]
                self.loss+=error**2
                #update latent vectors
                p = self.P[u].copy()
                q = self.Q[i].copy()
                bu = self.Bu[u]
                bi = self.Bi[i]
                self.P[u] = p+self.lRate*(error*q-self.regU*p)
                self.Q[i] = q+self.lRate*(error*p-self.regI*q)
                self.Bu[u] = bu+self.lRate*(error-self.regB*bu)
                self.Bi[i] = bi+self.lRate*(error-self.regB*bi)
            iteration += 1
            self.isConverged(iteration)

    def predict(self,u,i):
        if self.dao.containsUser(u) and self.dao.containsItem(i):
            return self.P[self.dao.user[u]].dot(self.Q[self.dao.item[i]])+self.dao.globalMean+self.Bi[self.dao.item[i]]+self.Bu[self.dao.user[u]]
        else:
            return self.dao.globalMean
