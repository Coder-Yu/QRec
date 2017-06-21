from baseclass.IterativeRecommender import IterativeRecommender
from baseclass.SocialRecommender import SocialRecommender
import numpy as np
from tool import config
class SocialMF(SocialRecommender ):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(SocialMF, self).__init__(conf,trainingSet,testSet,relation,fold)

    def readConfiguration(self):
        super(SocialMF, self).readConfiguration()

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                user, item, rating = entry
                u = self.dao.user[user]
                i = self.dao.item[item]
                error = rating - self.P[u].dot(self.Q[i])
                self.loss += error**2
                p = self.P[u].copy()
                q = self.Q[i].copy()
                self.P[u] += self.lRate * error * q
                self.Q[i] += self.lRate * (error * p - self.regI * q)

            for user in self.dao.user:
                fPred = 0
                denom = 0
                u = self.dao.user[user]
                relationLoss = np.zeros(self.k)
                followees = self.sao.getFollowees(user)
                for followee in followees:
                    weight= followees[followee]
                    if self.dao.containsUser(followee):
                        uf = self.dao.user[followee]
                        fPred += weight * self.P[uf]
                        denom += weight
                if denom <> 0:
                    relationLoss = p - fPred / denom

                self.loss +=  self.regS *  relationLoss.dot(relationLoss)

                # update latent vectors
                self.P[u] -= self.lRate * self.regS * relationLoss


            self.loss+=self.penaltyLoss()
            iteration += 1
            if self.isConverged(iteration):
                break

    def penaltyLoss(self):
        return self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum()