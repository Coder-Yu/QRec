from baseclass.SocialRecommender import SocialRecommender
import math
import numpy as np
from tool import config
from tool import qmath
from random import choice
from collections import defaultdict
class SBPR(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(SBPR, self).__init__(conf,trainingSet,testSet,relation,fold)
        self.userSocialItemsSetList = defaultdict(list)
        self.k = int(self.config['num.factors'])

    def readConfiguration(self):
        super(SBPR, self).readConfiguration()

    def initModel(self):
        super(SBPR, self).initModel()


        # find items rated by trusted neighbors only
        for userIdx in self.dao.user:
            userRatedItems = self.dao.trainSet_u[userIdx]
            if userRatedItems == None:
                continue   #user  userIdx have not rated items
            trustedUsers = self.sao.getFollowees(self.dao.getUserId(userIdx))
            items = []
            for trustedUser in trustedUsers:
                trustedRatedItems = self.dao.trainSet_u[trustedUser]
                for trustedRatedItemIdx in trustedRatedItems.keys():
                    if not (trustedRatedItemIdx in userRatedItems.keys())and not(trustedRatedItemIdx in items):
                        items.append(trustedRatedItemIdx)
            self.userSocialItemsSetList[userIdx] = items


    def printAlgorConfig(self):
        super(SBPR, self).printAlgorConfig()
        print 'Specified Arguments of', self.config['recommender'] + ':'
        print '=' * 80


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

                # social Items List
                socialItemsList = self.userSocialItemsSetList[userIdx]

                # negative item index
                while True:
                    negItemIdx = choice(self.dao.item.keys())
                    if  not(negItemIdx in ratedItems.keys()) and not(negItemIdx in socialItemsList):
                        break
                negPredictRating = self.predict(userIdx, negItemIdx)

                userId = self.dao.getUserId(userIdx)
                posItemId = self.dao.getItemId(posItemIdx)
                negItemId = self.dao.getItemId(negItemIdx)


                if len(socialItemsList) > 0:
                    socialItemIdx = choice(socialItemsList)
                    socialItemId = self.dao.getItemId(socialItemIdx)
                    socialPredictRating = self.predict(userIdx, socialItemIdx)


                    trustedUsers = self.sao.getFollowees(userId)

                    socialWeight = 0


                    for trustedUserIdx in trustedUsers:
                        socialRating = self.dao.rating(trustedUserIdx,socialItemIdx)
                        if socialRating > 0:
                            socialWeight += 1

                    posSocialDiffValue = (posPredictRating - socialPredictRating) / (1 + socialWeight)
                    socialNegDiffValue = socialPredictRating - negPredictRating
                    error = -math.log(qmath.sigmoid(posSocialDiffValue)) - math.log(qmath.sigmoid(socialNegDiffValue))
                    self.loss += error

                    posSocialGradient = qmath.sigmoid(-posSocialDiffValue)
                    socialNegGradient = qmath.sigmoid(-socialNegDiffValue)


                    # update P, Q
                    for factorIdx in range(self.k):
                        userFactorValue = self.P[userId][factorIdx]
                        posItemFactorValue = self.Q[posItemId][factorIdx]
                        socialItemFactorValue = self.Q[socialItemId][factorIdx]
                        negItemFactorValue = self.Q[negItemId][factorIdx]


                        delta_puf = posSocialGradient * (posItemFactorValue - socialItemFactorValue) / (1 + socialWeight)+ socialNegGradient * (socialItemFactorValue - negItemFactorValue)
                        self.P[userId][factorIdx] += self.lRate * (delta_puf - self.regU * userFactorValue)
                        self.Q[posItemId][factorIdx] += self.lRate * (posSocialGradient * userFactorValue / (1 + socialWeight) - self.regI  * posItemFactorValue)
                        delta_qkf = posSocialGradient * (-userFactorValue / (1 + socialWeight)) + socialNegGradient * userFactorValue
                        self.Q[socialItemId][factorIdx] += self.lRate * (delta_qkf - self.regI  * socialItemFactorValue)
                        self.Q[negItemId][factorIdx] += self.lRate * (socialNegGradient * (-userFactorValue) -self.regI  * negItemFactorValue)
                        self.loss += self.regU * userFactorValue * userFactorValue + self.regI  * posItemFactorValue * posItemFactorValue + self.regI  * negItemFactorValue * negItemFactorValue + self.regI  * socialItemFactorValue * socialItemFactorValue
                else:
                    #if no social neighbors, the same as BPR

                    posNegDiffValue = posPredictRating - negPredictRating
                    self.loss +=  -math.log(qmath.sigmoid(posNegDiffValue))
                    posNegGradient = qmath.sigmoid(-posNegDiffValue)



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
            predictRating = qmath.sigmoid(self.Q[i].dot(self.P[u]))
            return predictRating
        else:
            return qmath.sigmoid(self.dao.globalMean)

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            u = self.dao.getUserId(u)
            return self.Q.dot(self.P[u])
        else:
            return [self.dao.globalMean] * len(self.dao.item)


