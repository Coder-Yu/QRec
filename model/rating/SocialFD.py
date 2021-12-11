from base.socialRecommender import SocialRecommender
from util import config
import numpy as np

class SocialFD(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(SocialFD, self).__init__(conf,trainingSet,testSet,relation,fold)

    def initModel(self):
        super(SocialFD, self).initModel()
        self.Bu = np.random.rand(self.data.trainingSize()[0])/5 # biased value of user
        self.Bi = np.random.rand(self.data.trainingSize()[1])/5  # biased value of item
        self.H = np.random.rand(self.emb_size, self.emb_size) / 5
        self.P /= 10
        self.Q /= 10

    def readConfiguration(self):
        super(SocialFD, self).readConfiguration()
        eps = config.OptionConf(self.config['SocialFD'])
        self.alpha = float(eps['-alpha'])
        self.eta = float(eps['-eta'])
        self.beta = float(eps['-beta'])

    def trainModel(self):
        epoch = 0
        while epoch < self.maxEpoch:
            self.loss = 0
            for entry in self.data.trainingData:
                u, i, r = entry
                error = r - self.predictForRating(u, i)
                u = self.data.getUserId(u)
                i = self.data.getItemId(i)
                self.loss += error ** 2
                bu = self.Bu[u]
                bi = self.Bi[i]
                x = self.P[u]
                y = self.Q[i]
                d = (x - y).dot(self.H).dot(self.H.T).dot((x - y).T)
                derivative_d = self.H.dot((x - y).T.dot(x - y))
                if r > 0.7: #high ratings, ratings are compressed to range (0.01,1.01)
                    self.loss += self.eta*self.alpha * d
                    # update latent vectors
                    self.H -= self.lRate * ((error + self.eta*self.alpha) * derivative_d)
                    W = (self.H.dot(self.H.T) + self.H.dot(self.H.T).T)
                    self.P[u] -= self.lRate * ((error + self.eta*self.alpha) * (W.dot(np.array([x - y]).T)).T[0])
                    self.Q[i] += self.lRate * ((error + self.eta*self.alpha) * (W.dot(np.array([x - y]).T)).T[0])

                elif r <= 0.5: #low ratings
                    self.loss += self.eta*self.alpha * abs(1 - min(d, 1))
                    # update latent vectors
                    if d < 1:
                        self.H += self.lRate * ((self.eta*self.alpha - error) * derivative_d)
                    else:
                        self.H -= self.lRate * error * derivative_d
                    W = (self.H.dot(self.H.T) + self.H.dot(self.H.T).T)
                    if d < 1:
                        self.P[u] += self.lRate * ((-error + self.eta*self.alpha) * (W.dot(np.array([x - y]).T)).T[0])
                        self.Q[i] += self.lRate * ((error - self.eta*self.alpha) * (W.dot(np.array([x - y]).T)).T[0])
                    else:
                        self.P[u] += self.lRate * (-error * (W.dot(np.array([x - y]).T)).T[0])
                        self.Q[i] += self.lRate * (error * (W.dot(np.array([x - y]).T)).T[0])

                else: #medium
                    # update latent vectors
                    self.H -= self.lRate * ((error) * derivative_d)
                    W = (self.H.dot(self.H.T) + self.H.dot(self.H.T).T)
                    self.P[u] += self.lRate * ((-error) * (W.dot(np.array([x - y]).T)).T[0] - self.regU * x)
                    self.Q[i] += self.lRate * ((error) * (W.dot(np.array([x - y]).T)).T[0] - self.regI * y)
                self.Bu[u] += self.lRate * (error - self.regU * bu)
                self.Bi[i] += self.lRate * (error - self.regI * bi)
                self.P[u] += self.lRate*(error*x-self.regU*x)
                self.Q[i] += self.lRate*(error*y-self.regI*y)

            for user in self.data.user:
                relations = self.social.getFollowees(user)
                u = self.data.user[user]
                x = self.P[u]
                for followee in relations:
                    uf = self.data.getUserId(followee)
                    if uf != -1 and self.data.containsUser(followee):  # followee is in rating set
                        self.loss += (x - self.P[uf]).dot(self.H).dot(self.H.T).dot((x - self.P[uf]).T)
                        R = (self.H.dot(self.H.T) + self.H.T.dot(self.H))
                        derivative_s = self.H.dot((x - self.P[uf]).T.dot(x - self.P[uf]))
                        delta = R.dot(np.array([x - self.P[uf]]).T).T[0]
                        self.P[u] -= self.lRate * self.eta*self.beta * delta
                        self.H -= self.lRate * self.eta*self.beta * derivative_s
            epoch += 1
            self.loss+=self.regU*self.Bu.dot(self.Bu)+self.regI*self.Bi.dot(self.Bi)
            if self.isConverged(epoch):
                break

    def predictForRating(self, u, i):
        if self.data.containsUser(u) and self.data.containsItem(i):
            u = self.data.getUserId(u)
            i = self.data.getItemId(i)
            x = self.P[u]
            y = self.Q[i]
            d = (x-y).dot(self.H).dot(self.H.T).dot((x-y).T)
            return self.Bi[i] + self.Bu[u] +self.data.globalMean - d
        else:
            return self.data.globalMean


    def predictForRanking(self, u):
        if self.data.containsUser(u):
            u = self.data.user[u]
            x = self.P[u]
            res = np.array([0] * self.data.trainingSize()[1], dtype=float)
            A = self.H.dot(self.H.T)
            for i, y in enumerate(self.Q):
                res[i] = self.Bi[i] + self.Bu[u] +self.data.globalMean-((x - y).dot(A).dot((x - y).T))
            return res
        else:
            return np.array([self.data.globalMean] * len(self.data.item))


