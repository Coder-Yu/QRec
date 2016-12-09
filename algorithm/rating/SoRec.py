from baseclass.SocialRecommender import SocialRecommender
import math
import numpy as np
from tool import config
#Social Recommendation Using Probabilistic Matrix Factorization
class SoRec(SocialRecommender ):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(SoRec, self).__init__(conf,trainingSet,testSet,relation,fold)


    def readConfiguration(self):
        super(SoRec, self).readConfiguration()
        regular = config.LineConfig(self.config['reg.lambda'])
        self.regZ = float(regular['-z'])

    def buildModel(self):
        self.Z = np.random.rand(self.dao.trainingSize()[0], self.k)
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            #ratings
            for entry in self.dao.trainingData:
                u, i, r = entry
                error = r - self.predict(u, i)
                i = self.dao.getItemId(i)
                u = self.dao.getUserId(u)
                self.loss += error ** 2
                p = self.P[u].copy()
                q = self.Q[i].copy()
                self.loss += self.regU * p.dot(p) + self.regI * q.dot(q)
                # update latent vectors
                self.P[u] += self.lRate * (error * q - self.regU * p)
                self.Q[i] += self.lRate * (error * p - self.regI * q)

            #relations
            for entry in self.sao.relation:
                u, v, tuv = entry
                vminus = len(self.sao.getFollowers(v))# ~ d - (k)
                uplus = len(self.sao.getFollowees(u))#o~ d + (i)
                try:
                    weight = math.sqrt(vminus / (uplus + vminus + 0.0))
                except ZeroDivisionError:
                    weight = 1
                if u != -1 and self.dao.containsUser(u) and v != -1 and self.dao.containsUser(v):
                    v = self.dao.getUserId(v)
                    u = self.dao.getUserId(u)
                else:
                    continue
                euv =  weight * tuv - self.P[u].dot(self.Z[v]) # weight * tuv~ cik *
                self.loss +=self.regS * (euv ** 2)
                p = self.P[u].copy()
                z = self.Z[v].copy()
                self.loss += self.regZ * z.dot(z)
                # update latent vectors
                self.P[u] += self.lRate * (self.regS*euv * z)
                self.Z[v] += self.lRate * (euv * p - self.regZ * z)



            iteration += 1
            if self.isConverged(iteration):
                break
