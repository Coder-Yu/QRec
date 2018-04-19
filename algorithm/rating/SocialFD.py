from baseclass.SocialRecommender import SocialRecommender
from tool import config
import numpy as np



class SocialFD(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(SocialFD, self).__init__(conf,trainingSet,testSet,relation,fold)

    def initModel(self):
        super(SocialFD, self).initModel()
        self.Bu = np.random.rand(self.dao.trainingSize()[0])/2 # biased value of user
        self.Bi = np.random.rand(self.dao.trainingSize()[1])/2  # biased value of item
        self.H = np.random.rand(self.k, self.k)/2
        self.P /= 3
        self.Q /= 3


    def readConfiguration(self):
        super(SocialFD, self).readConfiguration()
        eps = config.LineConfig(self.config['SocialFD'])
        self.alpha = float(eps['-alpha'])
        self.eta = float(eps['-eta'])


    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                u, i, r = entry
                error = r - self.predict(u, i)
                u = self.dao.getUserId(u)
                i = self.dao.getItemId(i)
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

                    self.loss += self.eta*(1-self.alpha) * abs(1 - min(d, 1))
                    # update latent vectors
                    if d < 1:
                        self.H += self.lRate * ((self.eta*(1-self.alpha) - error) * derivative_d)
                    else:
                        self.H -= self.lRate * error * derivative_d
                    W = (self.H.dot(self.H.T) + self.H.dot(self.H.T).T)
                    if d < 1:
                        self.P[u] += self.lRate * ((-error + self.eta*(1-self.alpha)) * (W.dot(np.array([x - y]).T)).T[0])
                        self.Q[i] += self.lRate * ((error - self.eta*(1-self.alpha)) * (W.dot(np.array([x - y]).T)).T[0])
                    else:
                        self.P[u] += self.lRate * (-error * (W.dot(np.array([x - y]).T)).T[0])
                        self.Q[i] += self.lRate * (error * (W.dot(np.array([x - y]).T)).T[0])
                        # self.loss += self.regU * bu ** 2 + self.regI * bi ** 2 + self.regU * x.dot(x) + self.regI * y.dot(
                        #     y)
                        # # update latent vectors
                        # self.Bu[u] += self.lRate * (error - self.regU * bu)
                        # self.Bi[i] += self.lRate * (error - self.regI * bi)
                        # self.H += self.lRate * ((error ) * derivative_d)
                        # W = (self.H.dot(self.H.T) + self.H.T.dot(self.H))
                        # self.P[u] += self.lRate * (
                        # (error) * (W.dot(np.array([x - y]).T)).T[0] - self.regU * x)
                        # self.Q[i] += self.lRate * (
                        # (-error ) * (W.dot(np.array([x - y]).T)).T[0] - self.regI * y)

                else: #medium
                    # update latent vectors
                    self.H -= self.lRate * ((error) * derivative_d)
                    W = (self.H.dot(self.H.T) + self.H.dot(self.H.T).T)
                    self.P[u] += self.lRate * ((-error) * (W.dot(np.array([x - y]).T)).T[0] - self.regU * x)
                    self.Q[i] += self.lRate * ((error) * (W.dot(np.array([x - y]).T)).T[0] - self.regI * y)

                self.Bu[u] += self.lRate * (error - self.regU * bu)
                self.Bi[i] += self.lRate * (error - self.regI * bi)


            for user in self.dao.user:
                relations = self.sao.getFollowees(user)
                u = self.dao.user[user]
                x = self.P[u]
                for followee in relations:
                    uf = self.dao.getUserId(followee)
                    if uf <> -1 and self.dao.containsUser(followee):  # followee is in rating set
                        self.loss += (x - self.P[uf]).dot(self.H).dot(self.H.T).dot((x - self.P[uf]).T)
                        R = (self.H.dot(self.H.T) + self.H.T.dot(self.H))
                        derivative_s = self.H.dot((x - self.P[uf]).T.dot(x - self.P[uf]))
                        delta = R.dot(np.array([x - self.P[uf]]).T).T[0]
                        self.P[u] -= self.lRate * self.eta*self.alpha * delta
                        self.H -= self.lRate * self.eta*self.alpha * derivative_s
            iteration += 1
            self.loss+=self.regU*self.Bu.dot(self.Bu)+self.regI*self.Bi.dot(self.Bi)
            if self.isConverged(iteration):
                break

    def predict(self,u,i):
        if self.dao.containsUser(u) and self.dao.containsItem(i):
            u = self.dao.getUserId(u)
            i = self.dao.getItemId(i)
            x = self.P[u]
            y = self.Q[i]
            d = (x-y).dot(self.H).dot(self.H.T).dot((x-y).T)
            return self.Bi[i] + self.Bu[u] +self.dao.globalMean - d
        else:
            return self.dao.globalMean


    def predictForRanking(self, u):
        if self.dao.containsUser(u):
            u = self.dao.user[u]
            x = self.P[u]
            res = np.array([0] * self.dao.trainingSize()[1], dtype=float)
            A = self.H.dot(self.H.T)
            for i, y in enumerate(self.Q):
                res[i] = self.Bi[i] + self.Bu[u] +self.dao.globalMean-((x - y).dot(A).dot((x - y).T))
            return res
        else:
            return np.array([self.dao.globalMean] * len(self.dao.item))


