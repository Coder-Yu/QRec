#coding:utf8
from baseclass.IterativeRecommender import IterativeRecommender
import math
import numpy as np
from tool import qmath
from random import choice
from tool.qmath import sigmoid
from math import log
from collections import defaultdict
class BPR(IterativeRecommender):

    # BPR：Bayesian Personalized Ranking from Implicit Feedback
    # Steffen Rendle,Christoph Freudenthaler,Zeno Gantner and Lars Schmidt-Thieme

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(BPR, self).__init__(conf,trainingSet,testSet,fold)

    # def readConfiguration(self):
    #     super(BPR, self).readConfiguration()

    def initModel(self):
        super(BPR, self).initModel()


    def buildModel(self):

        print 'Preparing item sets...'
        self.PositiveSet = defaultdict(dict)
        #self.NegativeSet = defaultdict(list)

        for user in self.dao.user:
            for item in self.dao.trainSet_u[user]:
                if self.dao.trainSet_u[user][item] >= 1:
                    self.PositiveSet[user][item] = 1
                # else:
                #     self.NegativeSet[user].append(item)
        print 'training...'
        iteration = 0
        itemList = self.dao.item.keys()
        while iteration < self.maxIter:
            self.loss = 0
            for user in self.PositiveSet:
                u = self.dao.user[user]
                for item in self.PositiveSet[user]:
                    i = self.dao.item[item]
                    # if len(self.NegativeSet[user]) > 0:
                    #     item_j = choice(self.NegativeSet[user])
                    # else:
                    item_j = choice(itemList)
                    while (self.PositiveSet[user].has_key(item_j)):
                        item_j = choice(itemList)
                    j = self.dao.item[item_j]
                    s = sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))
                    self.P[u] += self.lRate * (1 - s) * (self.Q[i] - self.Q[j])
                    self.Q[i] += self.lRate * (1 - s) * self.P[u]
                    self.Q[j] -= self.lRate * (1 - s) * self.P[u]

                    self.P[u] -= self.lRate * self.regU * self.P[u]
                    self.Q[i] -= self.lRate * self.regI * self.Q[i]
                    self.Q[j] -= self.lRate * self.regI * self.Q[j]
                    self.loss += -log(s)
            self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()
            iteration += 1
            if self.isConverged(iteration):
                break
            # for sample in range(len(self.dao.user)):
            #     while True:
            #         userIdx = choice(self.dao.user.keys())
            #         ratedItems = self.dao.trainSet_u[userIdx]
            #         if len(ratedItems) != 0:
            #             break
            #
            #     #positive item index
            #     posItemIdx = choice(ratedItems.keys())
            #     posPredictRating = self.predict(userIdx, posItemIdx)
            #
            #     # negative item index
            #     while True:
            #         negItemIdx = choice(self.dao.item.keys())
            #         if  not(negItemIdx in ratedItems.keys()):
            #             break
            #     negPredictRating = self.predict(userIdx, negItemIdx)
            #
            #     userId = self.dao.getUserId(userIdx)
            #     posItemId = self.dao.getItemId(posItemIdx)
            #     negItemId = self.dao.getItemId(negItemIdx)
            #
            #
            #     posNegDiffValue = posPredictRating - negPredictRating
            #     self.loss +=  -math.log(qmath.sigmoid(posNegDiffValue))
            #     posNegGradient = qmath.sigmoid(-posNegDiffValue)
            #
            #     #update user factors, item factors
            #     for factorIdx in range(self.k):
            #         userFactorValue = self.P[self.dao.getUserId(userIdx)][factorIdx]
            #         posItemFactorValue = self.Q[self.dao.getItemId(posItemIdx)][factorIdx]
            #         negItemFactorValue = self.Q[self.dao.getItemId(negItemIdx)][factorIdx]
            #         self.P[userId][factorIdx] += self.lRate * (posNegGradient * (posItemFactorValue - negItemFactorValue) - self.regU * userFactorValue)
            #         self.Q[posItemId][factorIdx] += self.lRate * (posNegGradient * userFactorValue - self.regI  * posItemFactorValue)
            #         self.Q[negItemId][factorIdx] +=  self.lRate * (posNegGradient * (-userFactorValue) - self.regI  * negItemFactorValue)
            #         self.loss += self.regU * userFactorValue * userFactorValue + self.regI * posItemFactorValue * posItemFactorValue +  self.regI  * negItemFactorValue * negItemFactorValue
            #
            #
            # iteration += 1
            # if self.isConverged(iteration):
            #     break


    def predict(self,user,item):

        if self.dao.containsUser(user) and self.dao.containsItem(item):
            u = self.dao.getUserId(user)
            i = self.dao.getItemId(item)
            predictRating = self.Q[i].dot(self.P[u])
            return predictRating
        else:
            return self.dao.globalMean

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            u = self.dao.getUserId(u)
            return self.Q.dot(self.P[u])
        else:
            return [self.dao.globalMean] * len(self.dao.item)


