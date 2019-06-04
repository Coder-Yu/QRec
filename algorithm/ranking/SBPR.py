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


    def buildModel(self):
        self.b = np.random.random(self.data.trainingSize()[1])
        print 'Preparing item sets...'
        self.PositiveSet = defaultdict(dict)
        self.FPSet = defaultdict(dict)
        # self.NegativeSet = defaultdict(list)

        for user in self.data.user:
            for item in self.data.trainSet_u[user]:
                if self.data.trainSet_u[user][item] >= 1:
                    self.PositiveSet[user][item] = 1
                    # else:
                    #     self.NegativeSet[user].append(item)
            if self.social.user.has_key(user):
                for friend in self.social.getFollowees(user):
                    if self.data.user.has_key(friend):
                        for item in self.data.trainSet_u[friend]:
                            if not self.PositiveSet[user].has_key(item):
                                if not self.FPSet[user].has_key(item):
                                    self.FPSet[user][item] = 1
                                else:
                                    self.FPSet[user][item] += 1
        Suk=1
        print 'Training...'
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            itemList = self.data.item.keys()
            for user in self.PositiveSet:
                u = self.data.user[user]
                kItems = self.FPSet[user].keys()
                for item in self.PositiveSet[user]:
                    i = self.data.item[item]
                    for n in range(3): #negative sampling for 3 times
                        if len(self.FPSet[user]) > 0:
                            item_k = choice(kItems)
                            k = self.data.item[item_k]
                            s = sigmoid((self.P[u].dot(self.Q[i])+self.b[i] - self.P[u].dot(self.Q[k])-self.b[k])/ (Suk+1))
                            self.P[u] += 1 / (Suk+1) *self.lRate * (1 - s) * (self.Q[i] - self.Q[k])
                            self.Q[i] += 1 / (Suk+1) *self.lRate * (1 - s) * self.P[u]
                            self.Q[k] -= 1 / (Suk+1) *self.lRate * (1 - s) * self.P[u]
                            self.b[i] += 1 / (Suk+1) *self.lRate * (1 - s)
                            self.b[k] -= 1 / (Suk+1) *self.lRate * (1 - s)
                            item_j = ''
                            # if len(self.NegativeSet[user])>0:
                            #     item_j = choice(self.NegativeSet[user])
                            # else:
                            item_j = choice(itemList)
                            while (self.PositiveSet[user].has_key(item_j) or self.FPSet.has_key(item_j)):
                                item_j = choice(itemList)
                            j = self.data.item[item_j]
                            s = sigmoid(self.P[u].dot(self.Q[k])+self.b[k] - self.P[u].dot(self.Q[j])-self.b[j])
                            self.P[u] +=  self.lRate * (1 - s) * (self.Q[k] - self.Q[j])
                            self.Q[k] += self.lRate * (1 - s) * self.P[u]
                            self.Q[j] -= self.lRate * (1 - s) * self.P[u]
                            self.b[k] += self.lRate * (1 - s)
                            self.b[j] -= self.lRate * (1 - s)

                            self.P[u] -= self.lRate * self.regU * self.P[u]
                            self.Q[i] -= self.lRate * self.regI * self.Q[i]
                            self.Q[j] -= self.lRate * self.regI * self.Q[j]
                            self.Q[k] -= self.lRate * self.regI * self.Q[k]

                            self.loss += -log(sigmoid((self.P[u].dot(self.Q[i])+self.b[i] - self.P[u].dot(self.Q[k])-self.b[k])/ (Suk+1))) \
                                         - log(sigmoid(self.P[u].dot(self.Q[k])+self.b[k] - self.P[u].dot(self.Q[j])-self.b[j]))
                        else:
                            item_j = choice(itemList)
                            while (self.PositiveSet[user].has_key(item_j)):
                                item_j = choice(itemList)
                            j = self.data.item[item_j]
                            s = sigmoid(self.P[u].dot(self.Q[i])+self.b[i] - self.P[u].dot(self.Q[j])-self.b[j])
                            self.P[u] += self.lRate * (1 - s) * (self.Q[i] - self.Q[j])
                            self.Q[i] += self.lRate * (1 - s) * self.P[u]
                            self.Q[j] -= self.lRate * (1 - s) * self.P[u]
                            self.b[i] += self.lRate * (1 - s)
                            self.b[j] -= self.lRate * (1 - s)

                            self.loss += -log(s)

                self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()+self.b.dot(self.b)
            iteration += 1
            if self.isConverged(iteration):
                break


    def predict(self,user,item):

        if self.data.containsUser(user) and self.data.containsItem(item):
            u = self.data.getUserId(user)
            i = self.data.getItemId(item)
            predictRating = sigmoid(self.Q[i].dot(self.P[u])+self.b[i])
            return predictRating
        else:
            return sigmoid(self.data.globalMean)


    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.Q.dot(self.P[u])+self.b
        else:
            return [self.data.globalMean] * self.num_items


