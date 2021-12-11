from base.socialRecommender import SocialRecommender
import math
import numpy as np
from util import config
#Social Recommendation Using Probabilistic Matrix Factorization
class SoRec(SocialRecommender ):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(SoRec, self).__init__(conf,trainingSet,testSet,relation,fold)

    def readConfiguration(self):
        super(SoRec, self).readConfiguration()
        regZ = config.OptionConf(self.config['SoRec'])
        self.regZ = float( regZ['-z'])

    def initModel(self):
        super(SoRec, self).initModel()
        self.Z = np.random.rand(self.data.trainingSize()[0], self.emb_size) / 10

    def printAlgorConfig(self):
        super(SoRec, self).printAlgorConfig()
        print('Specified Arguments of', self.config['model.name'] + ':')
        print('regZ: %.3f' % self.regZ)
        print('=' * 80)

    def trainModel(self):
        epoch = 0
        while epoch < self.maxEpoch:
            self.loss = 0
            #ratings
            for entry in self.data.trainingData:
                user, item, rating = entry
                error = rating - self.predictForRating(user, item)
                i = self.data.item[item]
                u = self.data.user[user]
                self.loss += error ** 2
                p = self.P[u]
                q = self.Q[i]
                # update latent vectors
                self.P[u] += self.lRate * (error * q - self.regU * p)
                self.Q[i] += self.lRate * (error * p - self.regI * q)
            #relations
            for entry in self.social.relation:
                u, v, tuv = entry
                if self.data.containsUser(u) and self.data.containsUser(v):
                    vminus = len(self.social.getFollowers(v))# ~ d - (k)
                    uplus = len(self.social.getFollowees(u))#~ d + (i)
                    try:
                        weight = math.sqrt(vminus / (uplus + vminus + 0.0))
                    except ZeroDivisionError:
                        weight = 1
                    v = self.data.user[v]
                    u = self.data.user[u]
                    euv = weight * tuv - self.P[u].dot(self.Z[v])  # weight * tuv~ cik *
                    self.loss += self.regS * (euv ** 2)
                    p = self.P[u]
                    z = self.Z[v]

                    # update latent vectors
                    self.P[u] += self.lRate * (self.regS * euv * z)
                    self.Z[v] += self.lRate * (self.regS * euv * p - self.regZ * z)

            self.loss += self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum() + self.regZ*(self.Z*self.Z).sum()
            epoch += 1
            if self.isConverged(epoch):
                break

