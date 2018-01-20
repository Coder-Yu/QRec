from baseclass.IterativeRecommender import IterativeRecommender
import numpy as np

class SVD(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(SVD, self).__init__(conf,trainingSet,testSet,fold)

    def initModel(self):
        super(SVD, self).initModel()
        self.Bu = np.random.rand(self.dao.trainingSize()[0])/5  # bias value of user
        self.Bi = np.random.rand(self.dao.trainingSize()[1])/5  # bias value of item

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                user, item, rating = entry
                u = self.dao.user[user]
                i = self.dao.item[item]
                error = rating-self.predict(user,item)
                self.loss+=error**2
                p = self.P[u]
                q = self.Q[i]

                bu = self.Bu[u]
                bi = self.Bi[i]

                #update latent vectors
                self.P[u] += self.lRate*(error*q-self.regU*p)
                self.Q[i] += self.lRate*(error*p-self.regI*q)
                self.Bu[u] += self.lRate*(error-self.regB*bu)
                self.Bi[i] += self.lRate*(error-self.regB*bi)
            self.loss += self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum()\
               +self.regB*((self.Bu*self.Bu).sum()+(self.Bi*self.Bi).sum())
            iteration += 1
            self.isConverged(iteration)

    def predict(self,u,i):
        if self.dao.containsUser(u) and self.dao.containsItem(i):
            u = self.dao.user[u]
            i = self.dao.item[i]
            return self.P[u].dot(self.Q[i])+self.dao.globalMean+self.Bi[i]+self.Bu[u]
        else:
            return self.dao.globalMean

    def predictForRanking(self,u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            u = self.dao.getUserId(u)
            return self.Q.dot(self.P[u])+self.dao.globalMean + self.Bi + self.Bu[u]
        else:
            return [self.dao.globalMean] * len(self.dao.item)

