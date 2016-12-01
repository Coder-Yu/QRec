#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : TrustMF.py

from baseclass.SocialRecommender import SocialRecommender
import numpy as np
from tool import config

class TrustMF(SocialRecommender):
    def __init__(self,conf):
        super(TrustMF, self).__init__(conf)

    def initModel(self):
        super(TrustMF, self).initModel()

    def readConfiguration(self):
        super(SocialRecommender, self).readConfiguration()
        regular = config.LineConfig(self.config['reg.lambda'])
        self.regB=float(regular['-b'])
        self.regT=float(regular['-t'])

    def buildModel(self):
        self.Br = np.random.rand(self.dao.trainingSize()[0], self.k)  # latent user matrix
        self.Wr = np.random.rand(self.dao.trainingSize()[0], self.k)  # latent item matrix
        self.Vr = np.random.rand(self.dao.trainingSize()[1], self.k)  # latent item matrix
        self.trusterModel()
        # self.Be = np.random.rand(self.sao.trustSize()[0], self.k)  # latent user matrix
        # self.We = np.random.rand(self.sao.trustSize()[0], self.k)  # latent item matrix
        # self.Ve = np.random.rand(self.dao.trainingSize()[1], self.k)  # latent item matrix
        # self.trusteeModel()

    def trusterModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                u, i, r = entry
                mbu=len(self.sao.getFollowees(u))
                uid = self.dao.getUserId(u)
                iid = self.dao.getItemId(i)
                error = r-self.predict(uid,iid)
                nbu=len(self.dao.userRated(uid)[0])
                nvi=len(self.dao.itemRated(iid)[0])
                self.loss+=error**2+self.regB*((mbu+nbu)*self.Br[uid].dot(self.Br[uid])+nvi*self.Vr[iid].dot(self.Vr[iid]))
                self.Br[uid]=self.Br[uid]-self.lRate*(error*self.Vr[iid]+self.regB*(mbu+nbu)*self.Br[uid])
                self.Vr[iid]=self.Vr[iid]-self.lRate*(error*self.Br[uid]+self.regB*nvi*self.Vr[iid])

            for entry in self.sao.relation:
                user1,user2,weight=entry
                mwk=len(self.sao.getFollowers(user2))

                u1 = self.dao.user[user1]
                u2 = self.dao.user[user2]

                error = weight-self.Br[u1].dot(self.Wr[u2])
                self.loss+=self.regT*error**2+self.regB*mwk*self.Wr[u2].dot(self.Wr[u2])
                self.Br[u1]=self.Br[u1]-self.lRate*(self.regT*(self.Br[u1].dot(self.Wr[u2])-self.sao.elem(u1,u2))*self.Wr[u2])
                self.Wr[u2]=self.Wr[u2]-self.lRate*(self.regT*error*self.Br[u1]+self.regB*mwk*self.Wr[u2])

            iteration += 1
            self.isConverged(iteration)

    # def trusteeModel(self):
    #     iteration = 0
    #     while iteration < self.maxIter:
    #         self.loss = 0
    #         for entry in self.dao.trainingData:
    #             u, i, r = entry
    #             mwu=len(self.sao.getFollowers(u))
    #             u = self.dao.getUserId(u)
    #             i = self.dao.getItemId(i)
    #             error = r-self.predict2(u,i)
    #             nwu=len(self.dao.userRated(u)[0])
    #             nvi=len(self.dao.itemRated(i)[0])
    #             self.loss+=error**2+self.regB*((mwu+nwu)*self.We[u].dot(self.We[u])+nvi*self.Ve[i].dot(self.Ve[i]))
    #
    #             self.We[u]=self.We[u]-self.lRate*(error*self.Ve[i]+self.regB*(mwu+nwu)*self.We[u])
    #             self.Ve[i]=self.Ve[i]-self.lRate*(error*self.Be[u]+self.regB*nvi*self.Ve[i])
    #
    #         for entry in self.sao.relation:
    #             user1,user2,weight=entry
    #             mbk=len(self.sao.getFollowees(user2))
    #             u1 = self.dao.getUserId(user1)
    #             u2 = self.dao.getItemId(user2)
    #             error = weight-self.sao.B[u1].dot(self.sao.W[u2])
    #             self.loss+=self.regT*error**2+self.regB*mbk*self.Be[u1].dot(self.Be[u1])
    #             self.We[u]=self.We[u]-self.lRate*(error*self.Be[u1])
    #             self.Be[u1]=self.Be[u1]-self.lRate*(self.regT*error*self.dao.W[u2]+self.regB*mbk*self.Be[u1])
    #
    #         iteration += 1
    #         self.isConverged(iteration)

    def predict(self,u,i):
        if self.dao.containsUser(u) and self.dao.containsItem(i):
            u = self.dao.getUserId(u)
            i = self.dao.getItemId(i)
            return self.Br[u].dot(self.Vr[i])
        else:
            return self.dao.globalMean

    def predict1(self,u,i):
        if self.dao.containsUser(u) and self.dao.containsItem(i):
            u = self.dao.getUserId(u)
            i = self.dao.getItemId(i)
            return self.Br[u].dot(self.Vr[i])
        else:
            return self.dao.globalMean

    # def predict2(self,u,i):
    #     if self.dao.containsUser(u) and self.dao.containsItem(i):
    #         u = self.dao.getUserId(u)
    #         i = self.dao.getItemId(i)
    #         return self.We[u].dot(self.Ve[i])
    #     else:
    #         return self.dao.globalMean
    # def predict3(self,u1,u2):
    #     if self.sao.user.has_key(u1) and self.sao.user.has_key(u2):
    #         u1 = self.sao.user[u1]
    #         u2 = self.sao.user[u2]
    #         return self.Be[u1].dot(self.Ve[u2])
    #     else:
    #         return 0
