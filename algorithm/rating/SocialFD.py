from baseclass.IterativeRecommender import IterativeRecommender
from baseclass.SocialRecommender import SocialRecommender
from random import shuffle
from collections import defaultdict
from tool.qmath import denormalize,normalize
from tool.qmath import l2
from tool import config
import numpy as np
from math import sqrt



class SocialFD(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=[],fold='[1]'):
        super(SocialFD, self).__init__(conf,trainingSet,testSet,relation,fold)

    def initModel(self):
        super(SocialFD, self).initModel()
        # self.userProfile = defaultdict(dict)
        # self.itemProfile = defaultdict(dict)
        self.Bu = np.random.rand(self.dao.trainingSize()[0])/2 # biased value of user
        self.Bi = np.random.rand(self.dao.trainingSize()[1])/2  # biased value of item
        self.H = np.random.rand(self.k, self.k)/2
        self.P /= 1
        self.Q /= 1


    def readConfiguration(self):
        super(SocialFD, self).readConfiguration()
        eps = config.LineConfig(self.config['SocialFD'])
        self.eps = float(eps['-eps'])
        self.beta = float(eps['-c'])
        self.theta = float(eps['-t'])


    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            sum1 = 0
            sum2 = 0
            N1,N2 =0,0
            for entry in self.dao.trainingData:
                u, i, r = entry
                error = r - self.predict(u, i)
                # x = np.array([self.vecs['u'+u]/10])
                # y = np.array([self.vecs['i'+i]/10])
                u = self.dao.getUserId(u)
                i = self.dao.getItemId(i)
                self.loss += error ** 2

                bu = self.Bu[u]
                bi = self.Bi[i]
                x = self.P[u]
                y = self.Q[i]
                d = (x - y).dot(self.H).dot(self.H.T).dot((x - y).T)
                derivative_d = self.H.dot((x - y).T.dot(x - y))
                # if d > 0.2:
                #     self.eps = 0.001
                #     d = 0.2
                #     error = self.dao.globalMean + self.Bi[i] + self.Bu[u] + 0.2
                #     pass
                #
                # else:
                #     self.eps *= 1.05
                if r > 0.7:
                    self.loss += self.regU * bu ** 2 + self.regI * bi ** 2 + self.eps * d

                    # update latent vectors
                    self.H -= self.lRate * ((error + self.eps) * derivative_d)
                    W = (self.H.dot(self.H.T) + self.H.dot(self.H.T).T)
                    self.P[u] -= self.lRate * ((error + self.eps) * (W.dot(np.array([x - y]).T)).T[0])
                    self.Q[i] += self.lRate * ((error + self.eps) * (W.dot(np.array([x - y]).T)).T[0])


                elif r <= 0.5:

                    self.loss += self.regU * bu ** 2 + self.regI * bi ** 2 + self.eps * abs(
                        self.beta - min(d, self.beta))
                    # update latent vectors
                    if d < self.beta:
                        self.H += self.lRate * ((self.eps - error) * derivative_d)
                    else:
                        self.H -= self.lRate * error * derivative_d
                    W = (self.H.dot(self.H.T) + self.H.dot(self.H.T).T)
                    if d < self.beta:
                        self.P[u] += self.lRate * ((-error + self.eps) * (W.dot(np.array([x - y]).T)).T[0])
                        self.Q[i] += self.lRate * ((error - self.eps) * (W.dot(np.array([x - y]).T)).T[0])
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

                else:
                    self.loss += self.regU * bu ** 2 + self.regI * bi ** 2 + self.regU * x.dot(x) + self.regI * y.dot(
                        y)
                    # update latent vectors
                    self.H -= self.lRate * ((error) * derivative_d)
                    W = (self.H.dot(self.H.T) + self.H.dot(self.H.T).T)
                    self.P[u] += self.lRate * ((-error) * (W.dot(np.array([x - y]).T)).T[0] - self.regU * x)
                    self.Q[i] += self.lRate * ((error) * (W.dot(np.array([x - y]).T)).T[0] - self.regI * y)

                self.Bu[u] += self.lRate * (error - self.regU * bu)
                self.Bi[i] += self.lRate * (error - self.regI * bi)


                # if iteration == self.maxIter-1:
                    # if r > 0.7:

                        # sum1 += (x-y).dot(self.H).dot(self.H.T).dot((x-y).T)
                        # N1 += 1
                    # if r <= 0.5:

                        # sum2 += (x-y).dot(self.H).dot(self.H.T).dot((x-y).T)
                        # N2 += 1
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
                        self.P[u] -= self.lRate * self.theta * delta
                        self.H -= self.lRate * self.theta * derivative_s
            iteration += 1
            # if iteration == self.maxIter:
                # print 'high rating', float(sum1) / N1
                # print 'low rating', float(sum2) / N2
                # print self.dao.globalMean
                # print self.Bu.sum() / len(self.Bu)
            #print self.H
            if self.isConverged(iteration):
                break

    def predict(self,u,i):
        if self.dao.containsUser(u) and self.dao.containsItem(i): #and 'u'+u in self.vecs and 'i'+i in self.vecs:
            # x = np.array([self.vecs['u'+u]/10])
            # y = np.array([self.vecs['i'+i]/10])
            u = self.dao.getUserId(u)
            i = self.dao.getItemId(i)
            x = self.P[u]
            y = self.Q[i]
            d = (x-y).dot(self.H).dot(self.H.T).dot((x-y).T)

            return self.Bi[i] + self.Bu[u] +self.dao.globalMean - d
        else:
            return self.dao.globalMean

    # def predictForRanking(self,u):
    #     if self.dao.containsUser(u):
    #         u = self.dao.user[u]
    #         x = self.P[u]
    #         res = np.array([0] * self.dao.trainingSize()[1], dtype=float)
    #         A = self.H.dot(self.H.T)
    #         for i, y in enumerate(self.Q):
    #             res[i] = (x-y).dot(A).dot((x-y).T)
    #         res = self.Bi + self.Bu[u] +self.dao.globalMean - res
    #         return res
    #     else:
    #         return np.array([self.dao.globalMean] * len(self.dao.item))

    def predictForRanking(self, u):
        if self.dao.containsUser(u):
            u = self.dao.user[u]
            x = self.P[u]
            res = np.array([0] * self.dao.trainingSize()[1], dtype=float)
            A = self.H.dot(self.H.T)
            for i, y in enumerate(self.Q):
                res[i] = 10000-((x - y).dot(A).dot((x - y).T))
            return res
        else:
            return np.array([self.dao.globalMean] * len(self.dao.item))


