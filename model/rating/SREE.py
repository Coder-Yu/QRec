from base.socialRecommender import SocialRecommender
import numpy as np
from util import config

###################################
#NOTE: WE CONSIDER THAT THE SOCIAL TERM IN THE RATING PREDICTION EQUATION SHOULD
#BE MOVED OUT. THE LOSS FUNCTION SHOULD BE (RATING ERROR)^2 + SOCIAL TERM + PENALTY TERMS
#THEREFORE, THE IMPLEMENTATION IS DIFFERENT FROM THE ORIGINAL PAPER.
###################################

class SREE(SocialRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=list(),fold='[1]'):
        super(SREE, self).__init__(conf, trainingSet, testSet, relation,fold)

    def readConfiguration(self):
        super(SREE, self).readConfiguration()
        par = config.OptionConf(self.config['SREE'])
        self.alpha = float(par['-alpha'])

    def initModel(self):
        super(SREE, self).initModel()
        self.Bu = np.random.rand(self.data.trainingSize()[0])/10  # bias value of user
        self.Bi = np.random.rand(self.data.trainingSize()[1])/10  # bias value of item
        # self.X = np.random.rand(self.data.trainingSize()[0], self.Dim)/10
        # self.Y = np.random.rand(self.data.trainingSize()[1], self.Dim)/10

    def trainModel(self):
        epoch = 0
        while epoch < self.maxEpoch:
            self.loss = 0
            for entry in self.data.trainingData:
                user, item, rating = entry
                error = rating - self.predictForRating(user, item)
                u = self.data.user[user]
                i = self.data.item[item]
                self.loss += error ** 2
                self.loss += self.regU * (self.P[u] - self.Q[i]).dot(self.P[u] - self.Q[i])
                bu = self.Bu[u]
                bi = self.Bi[i]
                #self.loss += self.regB * bu ** 2 + self.regB * bi ** 2
                # update latent vectors
                self.P[u] -= self.lRate * (error + self.regU) * (self.P[u] - self.Q[i])
                self.Q[i] += self.lRate * (error + self.regI) * (self.P[u] - self.Q[i])
                self.Bu[u] += self.lRate * (error - self.regB * bu)
                self.Bi[i] += self.lRate * (error - self.regB * bi)
            self.loss+=self.regB*(self.Bu*self.Bu).sum()+self.regB*(self.Bi*self.Bi).sum()

            for user in self.social.user:
                if self.data.containsUser(user):
                    u = self.data.user[user]
                    followees = self.social.getFollowees(user)
                    for friend in followees:
                        if self.data.containsUser(friend):
                            v = self.data.user[friend]
                            weight = followees[friend]
                            p = self.P[u]
                            z = self.P[v]
                            # update latent vectors
                            self.P[u] -= self.lRate * self.alpha*weight*(p-z)
                            self.loss += self.alpha*weight*(p-z).dot(p-z)

            epoch += 1
            self.isConverged(epoch)

    def predictForRating(self, u, i):
        if self.data.containsUser(u) and self.data.containsItem(i):
            u = self.data.user[u]
            i = self.data.item[i]
            return self.data.globalMean + self.Bi[i] + self.Bu[u] - (self.P[u] - self.Q[i]).dot(self.P[u] - self.Q[i])
        else:
            return self.data.globalMean

    def predictForRanking(self,u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.user[u]
            res = ((self.Q-self.P[u])*(self.Q-self.P[u])).sum(axis=1)+self.Bi+self.Bu[u]+self.data.globalMean
            return res
        else:
            return [self.data.globalMean]*len(self.data.item)

