from baseclass.IterativeRecommender import IterativeRecommender
from baseclass.SocialRecommender import SocialRecommender

from tool import config
class SocialMF(SocialRecommender ):
    def __init__(self,conf):
        super(SocialMF, self).__init__(conf)

    def readConfiguration(self):
        super(SocialMF, self).readConfiguration()

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            relationError = 0
            relationLoss = 0
            for entry in self.dao.trainingData:
                userId, itemId, r = entry
                followees = self.sao.getFollowers(userId)
                u = self.dao.getUserId(userId)
                i = self.dao.getItemId(itemId)
                error = r - self.P[u].dot(self.Q[i])
                self.loss += error**2
                p = self.P[u].copy()
                q = self.Q[i].copy()
                for followee in followees:
                    weight= followees[followee]
                    uf = self.dao.getUserId(followee)
                    if uf <> -1 and self.dao.containsUser(uf):
                        relationLoss += weight *self.P[uf]
                relationError = p - relationLoss
                self.loss += self.regU * p.dot(p) + self.regI * q.dot(q) + self.regS * (relationError .dot(relationError ))


                # update latent vectors
                self.P[u] += self.lRate * (error * q - self.regU * p - self.regS * relationError)
                self.Q[i] += self.lRate * (error * p - self.regI * q)


            iteration += 1
            if self.isConverged(iteration):
                break
