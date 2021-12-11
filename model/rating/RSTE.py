from base.socialRecommender import SocialRecommender
from util import config
import numpy as np
class RSTE(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(RSTE, self).__init__(conf,trainingSet,testSet,relation,fold)

    def readConfiguration(self):
        super(RSTE, self).readConfiguration()
        alpha = config.OptionConf(self.config['RSTE'])
        self.alpha = float(alpha['-alpha'])

    def printAlgorConfig(self):
        super(RSTE, self).printAlgorConfig()
        print('Specified Arguments of',self.config['model.name']+':')
        print('alpha: %.3f' %self.alpha)
        print('='*80)

    def initModel(self):
        super(RSTE, self).initModel()

    def trainModel(self):
        epoch = 0
        while epoch < self.maxEpoch:
            self.loss = 0
            for entry in self.data.trainingData:
                user, item, rating = entry
                error = rating - self.predictForRating(user, item)
                i = self.data.item[item]
                u = self.data.user[user]
                self.loss += error ** 2
                p = self.P[u]
                q = self.Q[i]
                # update latent vectors
                self.P[u] += self.lRate * (self.alpha*error * q - self.regU * p)
                self.Q[i] += self.lRate * (self.alpha*error * p - self.regI * q)
            self.loss+= self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum()
            epoch += 1
            self.isConverged(epoch)

    def predictForRating(self, u, i):
        if self.data.containsUser(u) and self.data.containsItem(i):   
            i = self.data.getItemId(i)
            fPred = 0
            relations = self.social.getFollowees(u)
            weights = []
            indexes = []
            for followee in relations:
                if  self.data.containsUser(followee):  # followee is in rating set
                    indexes.append(self.data.user[followee])
                    weights.append(relations[followee])
            weights = np.array(weights)
            indexes = np.array(indexes)
            denom = weights.sum()
            u = self.data.user[u]
            if denom != 0:
                fPred += weights.dot((self.P[indexes].dot(self.Q[i])))

                return self.alpha * self.P[u].dot(self.Q[i])+(1-self.alpha)*fPred / denom
            else:
                return self.P[u].dot(self.Q[i])

        else:
            return self.data.globalMean

    def predictForRanking(self,u):
        if self.data.containsUser(u):
            fPred = 0
            denom = 0
            relations = self.social.getFollowees(u)
            for followee in relations:
                weight = relations[followee]
                if self.data.containsUser(followee):  # followee is in rating set
                    uf = self.data.user[followee]
                    fPred += weight * self.Q.dot(self.P[uf])
                    denom += weight
            u = self.data.user[u]
            if denom != 0:
                return self.alpha * self.Q.dot(self.P[u]) + (1 - self.alpha) * fPred / denom
            else:
                return self.Q.dot(self.P[u])
        else:
            return [self.data.globalMean] * len(self.data.item)