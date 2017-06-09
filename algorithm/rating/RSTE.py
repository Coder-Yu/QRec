from baseclass.SocialRecommender import SocialRecommender
from tool import config
import numpy as np
class RSTE(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(RSTE, self).__init__(conf,trainingSet,testSet,relation,fold)

    def readConfiguration(self):
        super(RSTE, self).readConfiguration()
        alpha = config.LineConfig(self.config['RSTE'])
        self.alpha = float(alpha['-alpha'])

    def printAlgorConfig(self):
        super(RSTE, self).printAlgorConfig()
        print 'Specified Arguments of',self.config['recommender']+':'
        print 'alpha: %.3f' %self.alpha
        print '='*80

    def initModel(self):
        super(RSTE, self).initModel()

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                user, item, rating = entry
                error = rating - self.predict(user,item)
                i = self.dao.item[item]
                u = self.dao.user[user]
                self.loss += error ** 2
                p = self.P[u]
                q = self.Q[i]
                self.loss += self.regU * p.dot(p) + self.regI * q.dot(q)
                # update latent vectors
                self.P[u] += self.lRate * (self.alpha*error * q - self.regU * p)
                self.Q[i] += self.lRate * (self.alpha*error * p - self.regI * q)
            iteration += 1
            self.isConverged(iteration)

    def predict(self,u,i):
        if self.dao.containsUser(u) and self.dao.containsItem(i):   
            i = self.dao.getItemId(i)
            fPred = 0
            denom = 0
            relations = self.sao.getFollowees(u)
            for followee in relations:
                weight = relations[followee]

                if  self.dao.containsUser(followee):  # followee is in rating set
                    uf = self.dao.user[followee]
                    fPred += weight * (self.P[uf].dot(self.Q[i]))
                    denom += weight
            u = self.dao.user[u]
            if denom <> 0:
                return self.alpha * self.P[u].dot(self.Q[i])+(1-self.alpha)*fPred / denom
            else:
                return self.P[u].dot(self.Q[i])
        else:
            return self.dao.globalMean

    def predictForRanking(self,u):
        if self.dao.containsUser(u):
            fPred = 0
            denom = 0
            relations = self.sao.getFollowees(u)
            for followee in relations:
                weight = relations[followee]
                if self.dao.containsUser(followee):  # followee is in rating set
                    uf = self.dao.user[followee]
                    fPred += weight * self.Q.dot(self.P[uf])
                    denom += weight
            u = self.dao.user[u]
            if denom <> 0:
                return self.alpha * self.Q.dot(self.P[u]) + (1 - self.alpha) * fPred / denom
            else:
                return self.Q.dot(self.P[u])
        else:
            return [self.dao.globalMean] * len(self.dao.item)