from baseclass.SocialRecommender import SocialRecommender
from tool import config
import numpy as np
import networkx as nx
import math
from tool import qmath

class LOCABAL(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(LOCABAL, self).__init__(conf,trainingSet,testSet,relation,fold)

    def readConfiguration(self):
        super(LOCABAL, self).readConfiguration()
        alpha = config.LineConfig(self.config['LOCABAL'])
        self.alpha = float(alpha['-alpha'])

    def printAlgorConfig(self):
        super(LOCABAL, self).printAlgorConfig()
        print 'Specified Arguments of',self.config['recommender']+':'
        print 'alpha: %.3f' %self.alpha
        print '='*80

    def initModel(self):
        super(LOCABAL, self).initModel()
        self.H = np.random.rand(self.k,self.k)
        G = nx.DiGraph()
        for re in self.sao.relation:
            G.add_edge(re[0], re[1])
        pr = nx.pagerank(G, alpha=0.85)
        self.W = {}
        for uid in pr:
            self.W[uid] = 1/(1+math.log(pr[uid]))
        self.S = {}
        for line in self.sao.relation:
            userId1,userId2,weight = line
            #add relations to dict
            if not self.S.has_key(userId1):
                self.S[userId1] = {}
            self.S[userId1][userId2] = qmath.cosine(userId1,userId2)

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                u, i, r = entry
                error = r - self.predict(u,i)
                i = self.dao.getItemId(i)
                u = self.dao.getUserId(u)
                self.loss += error ** 2
                p = self.P[u].copy()
                q = self.Q[i].copy()
                h = self.H.copy()
                self.loss += self.regU * p.dot(p) + self.regI * q.dot(q)
                if self.S.has_key(u):
                    for k in self.S[u]:
                        J = self.S[u][k] - np.dot(np.dot(p[u].T,h),p[k])
                        self.T += J * np.dot(p[u],p[k].T)
                else :
                    self.T = 0

                # update latent vectors
                self.P[u] += self.lRate * (self.W[u]*error * q - self.regU * p)
                self.Q[i] += self.lRate * (self.W[u]*error * p - self.regI * q)
                self.H += self.lRate * (self.alpha * self.T)
            iteration += 1
            self.isConverged(iteration)

    # def predict(self,u,i):
    #     if self.dao.containsUser(u) and self.dao.containsItem(i):
    #         i = self.dao.getItemId(i)
    #         fPred = 0
    #         denom = 0
    #         relations = self.sao.getFollowees(u)
    #         for followee in relations:
    #             weight = relations[followee]
    #             uf = self.dao.getUserId(followee)
    #             if uf <> -1 and self.dao.containsUser(followee):  # followee is in rating set
    #                 fPred += weight * (self.P[uf].dot(self.Q[i]))
    #                 denom += weight
    #         u = self.dao.getUserId(u)
    #         if denom <> 0:
    #             return self.alpha * self.P[u].dot(self.Q[i])+(1-self.alpha)*fPred / denom
    #         else:
    #             return self.P[u].dot(self.Q[i])
    #     else:
    #         return self.dao.globalMean