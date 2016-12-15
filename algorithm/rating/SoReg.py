from baseclass.SocialRecommender import SocialRecommender
from tool import config
from tool import qmath
import math


class SoReg(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(SoReg, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(SoReg, self).readConfiguration()
        alpha = config.LineConfig(self.config['SoReg'])
        self.alpha = float(alpha['-alpha'])

    def printAlgorConfig(self):
        super(SoReg, self).printAlgorConfig()
        print 'Specified Arguments of',self.config['recommender']+':'
        print 'alpha: %.3f' %self.alpha
        print '='*80

    def initModel(self):
        super(SoReg, self).initModel()

    def Sim(self,u,v):
        uid = self.dao.getUserId(u)
        vid = self.dao.getUserId(v)
        uR, fR = qmath.common(self.dao.col(uid), self.dao.col(vid))
        diffSum = 0
        diffUSum = 0
        diffVSum = 0
        if len(uR) != 0:
            for n in range(len(uR)):
                diffU = uR[n] - self.dao.userMeans[u]
                diffV = fR[n] - self.dao.userMeans[v]
                diffSum += diffU * diffV
                diffUSum += diffU ** 2
                diffVSum += diffV ** 2
            if diffUSum * diffVSum != 0:
                sim = (diffSum / math.sqrt(diffUSum * diffVSum) + 1) / 2
            else:
                sim = 0
        else:
            sim = 0
        return sim


    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                u, i, r = entry
                if r != 0:
                    I = 1
                else:
                    I = 0
                uid = self.dao.getUserId(u)
                id = self.dao.getItemId(i)
                simSumf = 0
                simSumg = 0
                simSum2 = 0
                if self.sao.getFollowees(u):
                    for f in self.sao.getFollowees(u):
                        if self.dao.containsUser(f):
                            fid = self.dao.getUserId(f)
                            simSumf += self.Sim(u,f)*(self.P[uid]-self.P[fid])
                            simSum2 += self.Sim(u, f) * ((self.P[uid] - self.P[fid]).dot(self.P[uid] - self.P[fid]))
                        else:
                            continue
                else:
                    continue
                if self.sao.getFollowers(u):
                    for g in self.sao.getFollowers(u):
                        if self.dao.containsUser(g):
                            gid = self.dao.getUserId(g)
                            simSumg += self.Sim(u,g)*(self.P[uid]-self.P[gid])
                        else:
                            continue
                else:
                    continue
                error = I * (r - self.P[uid].dot(self.Q[id]))
                p = self.P[uid].copy()
                q = self.Q[id].copy()
                self.loss += error**2 + 0.5*self.alpha*simSum2 + self.regU * p.dot(p) + self.regI * q.dot(q)

                #update latent vectors
                self.P[uid] += self.lRate*error*q + self.regU * p + self.alpha*simSumf + self.alpha*simSumg
                self.Q[id] += self.lRate*error*p + self.regI * q



            iteration += 1
            if self.isConverged(iteration):
                break
