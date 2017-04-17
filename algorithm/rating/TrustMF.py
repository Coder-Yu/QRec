#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : TrustMF.py

from baseclass.SocialRecommender import SocialRecommender
import numpy as np
from tool import config


class TrustMF(SocialRecommender):

    def __init__(self, conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(TrustMF, self).__init__(conf,trainingSet,testSet,relation,fold)

    def initModel(self):
        super(TrustMF, self).initModel()
        self.Br = np.random.rand(self.dao.trainingSize()[0], self.k)/10  # truster-specific feature matrix in truster model
        self.Wr = np.random.rand(self.dao.trainingSize()[0], self.k)/10  # trustee-specific feature matrix in truster model
        self.Vr = np.random.rand(self.dao.trainingSize()[1], self.k)/10  # item-specific    feature matrix in truster model
        self.Be = np.random.rand(self.dao.trainingSize()[0], self.k)/10  # truster-specific feature matrix in trustee model
        self.We = np.random.rand(self.dao.trainingSize()[0], self.k)/10  # trustee-specific feature matrix in trustee model
        self.Ve = np.random.rand(self.dao.trainingSize()[1], self.k)/10  # item-specific    feature matrix in trustee model

    def readConfiguration(self):
        super(TrustMF, self).readConfiguration()
        regular = config.LineConfig(self.config['reg.lambda'])
        self.regB = float(regular['-b'])
        self.regT = float(regular['-t'])

    def printAlgorConfig(self):
        super(TrustMF,self).printAlgorConfig()
        print 'Regularization parameter:  regT %.3f' % self.regT
        print '=' * 80

    def buildModel(self):
        # If necessary, you can fix the parameter in ./config/Trust.conf
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            self.trusterModel()
            self.trusteeModel()
            iteration += 1
            self.isConverged(iteration)


    def trusterModel(self):
        for entry in self.dao.trainingData:
            user, item, rating = entry
            u = self.dao.user[user]
            j = self.dao.item[item]
            error = self.truserPredict(user, item) - rating
            mbu = len(self.sao.getFollowees(user))
            nbu = len(self.dao.userRated(user)[0])
            nvj = len(self.dao.itemRated(item)[0])
            self.loss += error**2 + self.regB * ((mbu + nbu) * self.Br[u].dot(self.Br[u]) + nvj * self.Vr[j].dot(self.Vr[j]))
            self.Br[u] = self.Br[u] - self.lRate * (error * self.Vr[j] + self.regB * (mbu + nbu) * self.Br[u])
            self.Vr[j] = self.Vr[j] - self.lRate * (error * self.Br[u] + self.regB * nvj * self.Vr[j])

        #relations
        for entry in self.sao.relation:
            u, k, tuv = entry
            if self.dao.containsUser(u) and self.dao.containsUser(k):
                mwk = len(self.sao.getFollowers(k))
                u = self.dao.user[u]
                k = self.dao.user[k]
                error1= self.Br[u].dot(self.Wr[k]) - tuv
                self.loss+=self.regT * error1**2 + self.regB * mwk * self.Wr[k].dot(self.Wr[k])
                self.Br[u] = self.Br[u] - self.lRate * (self.regT * error1 * self.Wr[k])
                self.Wr[k] = self.Wr[k] - self.lRate * (self.regT * error1 * self.Br[u] + self.regB * mwk * self.Wr[k])

    def trusteeModel(self):
        for entry in self.dao.trainingData:
            user, item, rating = entry
            u = self.dao.user[user]
            j = self.dao.item[item]
            error = self.truseePredict(user, item) - rating
            mwu = len(self.sao.getFollowers(user))
            nwu = len(self.dao.userRated(user)[0])
            nvj = len(self.dao.itemRated(item)[0])
            self.loss += error**2 + self.regB * ((mwu + nwu) * self.We[u].dot(self.We[u]) + nvj * self.Ve[j].dot(self.Ve[j]))
            self.We[u] = self.We[u] - self.lRate * (error * self.Ve[j] + self.regB * (mwu + nwu) * self.We[u])
            self.Ve[j] = self.Ve[j] - self.lRate * (error * self.We[u] + self.regB * nvj * self.Ve[j])

        #relations
        for entry in self.sao.relation:
            k, u, tuv = entry
            if self.dao.containsUser(k) and self.dao.containsUser(u):
                mbk = len(self.sao.getFollowees(k))
                u = self.dao.user[u]
                k = self.dao.user[k]
                error1= self.Be[k].dot(self.We[u]) - tuv
                self.loss+=self.regT*error1**2+self.regB*mbk*self.Be[k].dot(self.Be[k])
                self.We[u] = self.We[u] - self.lRate * (self.regT * error1 * self.Be[k])
                self.Be[k] = self.Be[k] - self.lRate * (self.regT * error1 * self.We[u] + self.regB * mbk * self.Be[k])


    def truserPredict(self, u, i):
        if self.dao.containsUser(u) and self.dao.containsItem(i):
            u = self.dao.user[u]
            i = self.dao.item[i]
            return self.Br[u].dot(self.Vr[i])
        else:
            return self.dao.globalMean

    def truseePredict(self, u, i):
        if self.dao.containsUser(u) and self.dao.containsItem(i):
            u = self.dao.user[u]
            i = self.dao.item[i]
            return self.We[u].dot(self.Ve[i])
        else:
            return self.dao.globalMean

    def predict(self, u, i):
        if self.dao.containsUser(u) and self.dao.containsItem(i):
            u = self.dao.user[u]
            i = self.dao.item[i]
            return (self.Br[u] + self.We[u]).dot(self.Vr[i] + self.Ve[i]) * 0.25
        else:
            return self.dao.globalMean

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            u = self.dao.user[u]
            return (self.Vr + self.Ve).dot(self.Br[u] + self.We[u]) * 0.25
        else:
            return np.array([self.dao.globalMean] * len(self.dao.item))
