from base.socialRecommender import SocialRecommender
from util import config
import numpy as np
import networkx as nx
import math
from util import qmath

class LOCABAL(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(LOCABAL, self).__init__(conf,trainingSet,testSet,relation,fold)

    def readConfiguration(self):
        super(LOCABAL, self).readConfiguration()
        alpha = config.OptionConf(self.config['LOCABAL'])
        self.alpha = float(alpha['-alpha'])

    def printAlgorConfig(self):
        super(LOCABAL, self).printAlgorConfig()
        print('Specified Arguments of',self.config['model.name']+':')
        print('alpha: %.3f' %self.alpha)
        print('='*80)

    def initModel(self):
        super(LOCABAL, self).initModel()
        self.H = np.random.rand(self.emb_size, self.emb_size)
        G = nx.DiGraph()
        for re in self.social.relation:
            G.add_edge(re[0], re[1])
        pr = nx.pagerank(G, alpha=0.85)
        pr = sorted(iter(pr.items()),key=lambda d:d[1],reverse=True)
        pr = [(u[0],ind+1) for ind,u in enumerate(pr)]
        self.W = {}
        for user in pr:
            self.W[user[0]] = 1/(1+math.log(user[1]))
        self.S = {}
        for line in self.social.relation:
            u1,u2,weight = line
            if self.data.containsUser(u1) and self.data.containsUser(u2):
                uvec1=self.data.trainSet_u[u1]
                uvec2=self.data.trainSet_u[u2]
            #add relations to dict
                if u1 not in self.S:
                    self.S[u1] = {}
                self.S[u1][u2] = qmath.cosine_sp(uvec1,uvec2)

    def trainModel(self):
        epoch = 0
        while epoch < self.maxEpoch:
            self.loss = 0
            for entry in self.data.trainingData:
                user, item, r = entry
                error = r - self.predictForRating(user, item)
                i = self.data.getItemId(item)
                u = self.data.getUserId(user)
                if user in self.W:
                    self.loss += self.W[user]*error ** 2
                else:
                    self.loss += error ** 2
                p = self.P[u].copy()
                q = self.Q[i].copy()
                #update latent vectors
                if user in self.W:
                    self.P[u] += self.lRate * (self.W[user]*error * q - self.regU * p)
                    self.Q[i] += self.lRate * (self.W[user]*error * p - self.regI * q)
                #else:
                self.P[u] += self.lRate * (error * q - self.regU * p)
                self.Q[i] += self.lRate * (error * p - self.regI * q)
            for user in self.S:
                for friend in self.S[user]:
                    k = self.data.getUserId(friend)
                    u = self.data.getUserId(user)
                    p = self.P[u].copy()
                    q = self.P[k].copy()
                    error = self.S[user][friend] - np.dot(np.dot(p,self.H),q)
                    self.loss+=self.alpha*error**2
                    #update latent vectors
                    self.H+=self.lRate*self.alpha*error*(p.reshape(self.emb_size, 1).dot(q.reshape(1, self.emb_size)))
                    self.H-=self.lRate*self.regS*self.H
                    self.P[u]+=self.lRate*self.alpha*error*(self.H.dot(q))
                    self.P[k]+=self.lRate*self.alpha*error*(p.T.dot(self.H))
            self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()+self.regS*(self.H*self.H).sum()
            epoch += 1
            self.isConverged(epoch)