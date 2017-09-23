#coding:utf8
from baseclass.IterativeRecommender import IterativeRecommender
import math
import numpy as np
from tool import qmath
from random import choice
class BPR(IterativeRecommender):

    # BPR：Bayesian Personalized Ranking from Implicit Feedback
    # Steffen Rendle,Christoph Freudenthaler,Zeno Gantner and Lars Schmidt-Thieme

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(BPR, self).__init__(conf,trainingSet,testSet,fold)

    # def readConfiguration(self):
    #     super(BPR, self).readConfiguration()

    def initModel(self):
        super(BPR, self).initModel()
        self.Bi  = np.random.rand(self.dao.trainingSize()[1]) /5 # bias value of item
        #self.k = int(self.config['num.factors'])


    # def printAlgorConfig(self):
    #     super(BPR, self).printAlgorConfig()
    #     print 'Specified Arguments of', self.config['recommender'] + ':'
    #     print '=' * 80


    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for sample in range(len(self.dao.user)):
                while True:
                    userIdx = choice(self.dao.user.keys())
                    ratedItems = self.dao.trainSet_u[userIdx]
                    if len(ratedItems) != 0:
                        break

                #positive item index
                posItemIdx = choice(ratedItems.keys())
                posPredictRating = self.predict(userIdx, posItemIdx)

                # negative item index
                while True:
                    negItemIdx = choice(self.dao.item.keys())
                    if  not(negItemIdx in ratedItems.keys()):
                        break
                negPredictRating = self.predict(userIdx, negItemIdx)

                userId = self.dao.getUserId(userIdx)
                posItemId = self.dao.getItemId(posItemIdx)
                negItemId = self.dao.getItemId(negItemIdx)


                posNegDiffValue = posPredictRating - negPredictRating
                self.loss +=  -math.log(qmath.sigmoid(posNegDiffValue))
                posNegGradient = qmath.sigmoid(-posNegDiffValue)

                # update bi, bj
                posItemBiasValue = self.Bi[posItemId]
                self.Bi[posItemId] += self.lRate * (posNegGradient - self.regB * posItemBiasValue)
                self.loss += self.regB * posItemBiasValue * posItemBiasValue
                negItemBiasValue = self.Bi[negItemId]
                self.Bi[negItemId]  += self.lRate * (-posNegGradient - self.regB * negItemBiasValue)
                self.loss += self.regB * negItemBiasValue * negItemBiasValue

                #update user factors, item factors
                for factorIdx in range(self.k):
                    userFactorValue = self.P[self.dao.getUserId(userIdx)][factorIdx]
                    posItemFactorValue = self.Q[self.dao.getItemId(posItemIdx)][factorIdx]
                    negItemFactorValue = self.Q[self.dao.getItemId(negItemIdx)][factorIdx]
                    self.P[userId][factorIdx] += self.lRate * (posNegGradient * (posItemFactorValue - negItemFactorValue) - self.regU * userFactorValue)
                    self.Q[posItemId][factorIdx] += self.lRate * (posNegGradient * userFactorValue - self.regI  * posItemFactorValue)
                    self.Q[negItemId][factorIdx] +=  self.lRate * (posNegGradient * (-userFactorValue) - self.regI  * negItemFactorValue)
                    self.loss += self.regU * userFactorValue * userFactorValue + self.regI * posItemFactorValue * posItemFactorValue +  self.regI  * negItemFactorValue * negItemFactorValue


            iteration += 1
            if self.isConverged(iteration):
                break


    def predict(self,user,item):

        if self.dao.containsUser(user) and self.dao.containsItem(item):
            u = self.dao.getUserId(user)
            i = self.dao.getItemId(item)
            predictRating = self.Q[i].dot(self.P[u]) + self.Bi[i]
            return predictRating
        else:
            return self.dao.globalMean

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            u = self.dao.getUserId(u)
            return self.Q.dot(self.P[u])+self.Bi
        else:
            return [self.dao.globalMean] * len(self.dao.item)


