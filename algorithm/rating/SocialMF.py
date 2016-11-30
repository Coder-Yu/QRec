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
            for entry in self.dao.trainingData:
                userId, itemId, r = entry
                #get userU's Followers and Followees
                followersU = self.sao.getFollowers(userId)
                followeesU = self.sao.getFollowees(userId)
                u = self.dao.getUserId(userId)
                i = self.dao.getItemId(itemId)
                error = r - self.P[u].dot(self.Q[i])
                self.loss += error**2
                p = self.P[u].copy()
                q = self.Q[i].copy()
                trustAffect = 0
                for followee in followeesU:
                    trustUV= followeesU[followee]
                    uf = self.dao.getUserId(followee)
                    if uf <> -1 and self.dao.containsUser(uf):
                        trustAffect += trustUV *self.P[uf]
                relationLoss = p - trustAffect
                self.loss += self.regU * p.dot(p) + self.regI * q.dot(q) + self.regB * (relationLoss .dot(relationLoss ))


                # update latent vectors
                indirect = 0
                relationAffect = 0
                for follower in followersU:
                    trustVU = followersU[follower]
                    uf2 = self.dao.getUserId(follower)
                    followeesV = self.sao.getFollowees(uf2)
                    for followee in followeesV:
                        vf = self.dao.getUserId(followee)
                        trustVW = followeesV[followee]
                        if vf <> -1 and self.dao.containsUser(vf):
                            indirect = trustVW *  self.P[vf]
                    if uf2<> -1 and self.dao.containsUser(uf2):
                        relationAffect = trustVU *(self.P[uf2] - indirect)
                self.P[u] += self.lRate * (error * q - self.regU * p - self.regS * relationLoss+self.regS*relationAffect)
                self.Q[i] += self.lRate * (error * p - self.regI * q)


            iteration += 1
            if self.isConverged(iteration):
                break
