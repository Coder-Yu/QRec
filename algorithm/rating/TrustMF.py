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
        self.Br = np.random.rand(self.dao.trainingSize()[0], self.k)  # latent user matrix
        self.Wr = np.random.rand(self.dao.trainingSize()[0], self.k)  # latent item matrix
        self.Vr = np.random.rand(self.dao.trainingSize()[1], self.k)  # latent item matrix
        self.Be = np.random.rand(self.dao.trainingSize()[0], self.k)  # latent user matrix
        self.We = np.random.rand(self.dao.trainingSize()[0], self.k)  # latent item matrix
        self.Ve = np.random.rand(self.dao.trainingSize()[1], self.k)  # latent item matrix

    def readConfiguration(self):
        super(TrustMF, self).readConfiguration()
        regular = config.LineConfig(self.config['reg.lambda'])
        self.regB = float(regular['-b'])
        self.regT = float(regular['-t'])

    def printAlgorConfig(self):
        print 'Reduced Dimension:', self.k
        print 'Maximum Iteration:', self.maxIter
        print 'Regularization parameter:  regB %.3f regT %.3f' % (self.regB, self.regT)
        print '=' * 80

    def buildModel(self):
        # If necessary, you can fix the parameter in ./config/Trust.conf
        self.trusterModel()
        # train trusterModel and trusteeModel independently using the same
        # parameter setting.
        learningrate = config.LineConfig(self.config['learnRate'])
        self.lRate = float(learningrate['-init'])
        self.trusteeModel()

    def trusterModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                u, i, r = entry
                mbu = len(self.sao.getFollowees(u))
                uid = self.dao.getUserId(u)
                iid = self.dao.getItemId(i)
                error = self.truserPredict(u, i) - r
                nbu = len(self.dao.userRated(u)[0])
                nvi = len(self.dao.itemRated(i)[0])
                self.loss += error**2 + self.regB * ((mbu + nbu) * self.Br[uid].dot(self.Br[uid]) + nvi * self.Vr[iid].dot(self.Vr[iid]))
                self.Vr[iid] = self.Vr[iid] - self.lRate * (error * self.Br[uid] + self.regB * nvi * self.Vr[iid])

                relations = self.sao.getFollowees(u)
                for followee in relations:
                    weight = relations[followee]
                    uf = self.dao.getUserId(followee)
                    if uf != -1 and self.dao.containsUser(uf):  # followee is in rating set
                        error1 = self.Br[uid].dot(self.Wr[uf]) - weight
                        mwk = len(self.sao.getFollowers(followee))
                        self.loss += self.regT * error1**2 + self.regB * mwk * self.Wr[uf].dot(self.Wr[uf])
                        self.Br[uid] = self.Br[uid] - self.lRate * (error * self.Vr[iid] + self.regB * (mbu + nbu) * self.Br[uid] + self.regT * (self.Br[uid].dot(self.Wr[uf]) - weight) * self.Wr[uf])
                        self.Wr[uf] = self.Wr[uf] - self.lRate * (self.regT * error1 * self.Br[u] + self.regB * mwk * self.Wr[uf])

            iteration += 1
            self.isConverged(iteration)

    def trusteeModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                u, i, r = entry
                mwu = len(self.sao.getFollowers(u))
                uid = self.dao.getUserId(u)
                iid = self.dao.getItemId(i)
                error = self.truseePredict(u, i) - r
                nwu = len(self.dao.userRated(u)[0])
                nvi = len(self.dao.itemRated(i)[0])
                self.loss += error**2 + self.regB * ((mwu + nwu) * self.We[uid].dot(self.We[uid]) + nvi * self.Ve[iid].dot(self.Ve[iid]))
                self.Ve[iid] = self.Ve[iid] - self.lRate * (error * self.We[uid] + self.regB * nvi * self.Ve[iid])

                relations = self.sao.getFollowers(u)
                for follower in relations:
                    weight = relations[follower]
                    uf = self.dao.getUserId(follower)
                    if uf != -1 and self.dao.containsUser(uf):  # follower is in rating set
                        error1 = self.Be[uf].dot(self.We[u]) - weight
                        mbk = len(self.sao.getFollowees(follower))
                        self.loss += self.regT * error1**2 + self.regB * mbk * self.Be[uf].dot(self.Be[uf])
                        self.We[u] = self.We[u] - self.lRate * (error * self.Vr[iid] + self.regB * (mwu + nwu) * self.We[uid] + self.regT * error1 * self.Be[uf])
                        self.Be[uf] = self.Be[uf] - self.lRate * (self.regT * error1 * self.We[uid] + self.regB * mbk * self.Be[uf])

            iteration += 1
            self.isConverged(iteration)

    def truserPredict(self, u, i):
        if self.dao.containsUser(u) and self.dao.containsItem(i):
            u = self.dao.getUserId(u)
            i = self.dao.getItemId(i)
            return self.Br[u].dot(self.Vr[i])
        else:
            return self.dao.globalMean

    def truseePredict(self, u, i):
        if self.dao.containsUser(u) and self.dao.containsItem(i):
            u = self.dao.getUserId(u)
            i = self.dao.getItemId(i)
            return self.We[u].dot(self.Ve[i])
        else:
            return self.dao.globalMean

    def predict(self, u, i):
        if self.dao.containsUser(u) and self.dao.containsItem(i):
            u = self.dao.getUserId(u)
            i = self.dao.getItemId(i)
            return (self.Br[u] + self.We[u]).dot(self.Vr[i] + self.Ve[i]) * 0.25
        else:
            return self.dao.globalMean