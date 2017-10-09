from baseclass.SocialRecommender import SocialRecommender
from math import log
import numpy as np
from tool import config
from tool.qmath import sigmoid
from random import choice
from collections import defaultdict
class SBPR(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(SBPR, self).__init__(conf,trainingSet,testSet,relation,fold)
        self.userSocialItemsSetList = defaultdict(list)


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
        print 'Preparing item sets...'
        self.PositiveSet = defaultdict(dict)
        self.IPositiveSet = defaultdict(dict)
        # self.NegativeSet = defaultdict(list)

        for user in self.dao.user:
            for item in self.dao.trainSet_u[user]:
                if self.dao.trainSet_u[user][item] >= 1:
                    self.PositiveSet[user][item] = 1
                    # else:
                    #     self.NegativeSet[user].append(item)
            if self.sao.user.has_key(user):
                for friend in self.sao.getFollowees(user):
                    if self.dao.user.has_key(friend):
                        for item in self.dao.trainSet_u[friend[0]]:
                            if not self.PositiveSet[user].has_key(item):
                                if not self.IPositiveSet[user].has_key(item):
                                    self.IPositiveSet[user][item] = 1
                                else:
                                    self.IPositiveSet[user][item] += 1

        print 'Training...'
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            itemList = self.dao.item.keys()
            for user in self.PositiveSet:
                u = self.dao.user[user]
                for item in self.PositiveSet[user]:
                    i = self.dao.item[item]
                    kItems = self.IPositiveSet[user].keys()
                    if len(self.IPositiveSet[user]) > 0:
                        item_k = choice(kItems)
                        k = self.dao.item[item_k]
                        Suk = self.IPositiveSet[user][item_k]
                        self.P[u] += (1 / (Suk+1)) *self.lRate * (1 - sigmoid((self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[k]))/ (Suk+1))) * (
                            self.Q[i] - self.Q[k])
                        self.Q[i] += (1 / (Suk+1)) *self.lRate * (1 - sigmoid((self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[k]))/ (Suk+1))) * \
                                     self.P[u]
                        self.Q[k] -= (1 / (Suk+1)) *self.lRate * (1 - sigmoid((self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[k]))/ (Suk+1))) * self.P[u]

                        item_j = ''
                        # if len(self.NegativeSet[user])>0:
                        #     item_j = choice(self.NegativeSet[user])
                        # else:
                        item_j = choice(itemList)
                        while (self.PositiveSet[user].has_key(item_j)):
                            item_j = choice(itemList)
                        j = self.dao.item[item_j]
                        self.P[u] +=  self.lRate * (1 - sigmoid((self.P[u].dot(self.Q[k]) - self.P[u].dot(self.Q[j])))) * (self.Q[k] - self.Q[j])
                        self.Q[k] += self.lRate * (1 - sigmoid((self.P[u].dot(self.Q[k]) - self.P[u].dot(self.Q[j])))) * self.P[u]
                        self.Q[j] -= self.lRate * (1 - sigmoid((self.P[u].dot(self.Q[k]) - self.P[u].dot(self.Q[j])))) * self.P[u]

                        self.P[u] -= self.lRate * self.regU * self.P[u]
                        self.Q[i] -= self.lRate * self.regI * self.Q[i]
                        self.Q[j] -= self.lRate * self.regI * self.Q[j]
                        self.Q[k] -= self.lRate * self.regI * self.Q[k]

                        self.loss += -log(sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[k]))) - \
                                     log(sigmoid(self.P[u].dot(self.Q[k]) - self.P[u].dot(self.Q[j])))
                    else:
                        item_j = choice(itemList)
                        while (self.PositiveSet[user].has_key(item_j)):
                            item_j = choice(itemList)
                        j = self.dao.item[item_j]
                        self.P[u] += self.lRate * (1 - sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))) * (
                            self.Q[i] - self.Q[j])
                        self.Q[i] += self.lRate * (1 - sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))) * \
                                     self.P[u]
                        self.Q[j] -= self.lRate * (1 - sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))) * \
                                     self.P[u]

                        self.loss += -log(sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j])))

            self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()
            iteration += 1
            if self.isConverged(iteration):
                break


    def predict(self,user,item):

        if self.dao.containsUser(user) and self.dao.containsItem(item):
            u = self.dao.getUserId(user)
            i = self.dao.getItemId(item)
            predictRating = sigmoid(self.Q[i].dot(self.P[u]))
            return predictRating
        else:
            return sigmoid(self.dao.globalMean)

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            u = self.dao.getUserId(u)
            return self.Q.dot(self.P[u])
        else:
            return [self.dao.globalMean] * len(self.dao.item)


