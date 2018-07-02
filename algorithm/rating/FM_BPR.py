from baseclass.SocialRecommender import SocialRecommender
from math import log
import numpy as np
from tool import config
from tool.qmath import sigmoid
from random import choice
from collections import defaultdict
class FM_BPR(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(FM_BPR, self).__init__(conf,trainingSet,testSet,relation,fold)

    def readConfiguration(self):
        super(FM_BPR, self).readConfiguration()
        options = config.LineConfig(self.config['FM_BPR'])
        self.lambda_f = float(options['-f'])
        self.lambda_m = float(options['-m'])

    def initModel(self):
        super(FM_BPR, self).initModel()
        print 'Kind Note: This method will probably take much time.'

        # data clean
        cleanList = []
        cleanPair = []
        for user in self.sao.followees:
            if not self.dao.user.has_key(user):
                cleanList.append(user)
            for u2 in self.sao.followees[user]:
                if not self.dao.user.has_key(u2):
                    cleanPair.append((user, u2))
        for u in cleanList:
            del self.sao.followees[u]

        for pair in cleanPair:
            if self.sao.followees.has_key(pair[0]):
                del self.sao.followees[pair[0]][pair[1]]

        cleanList = []
        cleanPair = []
        for user in self.sao.followers:
            if not self.dao.user.has_key(user):
                cleanList.append(user)
            for u2 in self.sao.followers[user]:
                if not self.dao.user.has_key(u2):
                    cleanPair.append((user, u2))
        for u in cleanList:
            del self.sao.followers[u]

        for pair in cleanPair:
            if self.sao.followers.has_key(pair[0]):
                del self.sao.followers[pair[0]][pair[1]]

        # build MNET,FNET
        print 'Building Membership and Friendship networks...'

        self.MNet = defaultdict(dict)

        for user1 in self.dao.trainSet_u:
            s1 = set(self.dao.trainSet_u[user1].keys())
            for user2 in self.dao.trainSet_u:
                s2 = set(self.dao.trainSet_u[user2].keys())
                if user1==user2:
                    continue
                weight=len(s1.intersection(s2))
                if weight>3:
                    self.MNet[user1][user2]=1+log(weight)

        self.FNet = defaultdict(dict)
        for user1 in self.sao.user:
            if self.dao.containsUser(user1):
                s1 = set(self.sao.followees[user1].keys())
                for user2 in s1:
                    if self.dao.containsUser(user2):
                        s2 = set(self.sao.followees[user2].keys())
                        weight = len(s1.intersection(s2))
                        if weight>1:
                            self.FNet[user1][user2]=1+log(weight)
                        else:
                            self.FNet[user1][user2] = 1


    def buildModel(self):
        super(FM_BPR, self).buildModel()
        self.M1 = np.random.random((self.k,self.k))/5
        self.M2 = np.random.random((self.k,self.k))/5
        self.Z = np.random.random((self.dao.trainingSize()[0],self.k))/10
        self.F = np.random.random((self.dao.trainingSize()[0],self.k))/10

        self.b = np.random.random(self.dao.trainingSize()[1])
        print 'Preparing item sets...'
        self.PositiveSet = defaultdict(dict)
        self.JointSet = defaultdict(dict)
        self.OkSet = defaultdict(dict)


        for user in self.dao.user:
            for item in self.dao.trainSet_u[user]:
                if self.dao.trainSet_u[user][item] >= 1:
                    self.PositiveSet[user][item] = 1
                    # else:
                    #     self.NegativeSet[user].append(item)

        for user in self.dao.user:
            if self.MNet.has_key(user) and self.FNet.has_key(user):
                for friend in self.MNet[user]:
                    if self.FNet[user].has_key(friend):
                        for item in self.PositiveSet[friend]:
                            self.JointSet[user][item]=1

        for user in self.dao.user:
            if self.MNet.has_key(user):
                for friend in self.MNet[user]:
                    for item in self.PositiveSet[friend]:
                        if not self.JointSet[user].has_key(item):
                            self.OkSet[user][item] = 1

        for user in self.dao.user:
            if self.FNet.has_key(user):
                for friend in self.FNet[user]:
                    for item in self.PositiveSet[friend]:
                        if not self.JointSet[user].has_key(item):
                            self.OkSet[user][item] = 1


        print 'Training...'
        iteration = 0
        self.s=1
        while iteration < self.maxIter:
            self.loss = 0
            itemList = self.dao.item.keys()

            for user in self.PositiveSet:
                jItems = self.JointSet[user].keys()
                kItems = self.OkSet[user].keys()
                u = self.dao.user[user]
                for item in self.PositiveSet[user]:
                    i = self.dao.item[item]

                    for ind in range(1):
                        if len(jItems) and len(kItems)>0:
                            item_j = choice(jItems)
                            j = self.dao.item[item_j]
                            self.optimization(u,i,j)
                            item_k = choice(kItems)
                            k = self.dao.item[item_k]
                            self.optimization(u, j, k)
                            item_n = choice(itemList)
                            while self.PositiveSet[user].has_key(item_n) or self.JointSet[user].has_key(item_n)\
                                or self.OkSet[user].has_key(item_n):
                                item_n = choice(itemList)
                            n = self.dao.item[item_n]
                            self.optimization(u,k,n)
                            # self.loss += -log(sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[k]))) - \
                            #              log(sigmoid(
                            #                  (1 / self.s) * (self.P[u].dot(self.Q[k]) - self.P[u].dot(self.Q[j]))))
                        elif len(kItems)>0:
                            item_k = choice(kItems)
                            k = self.dao.item[item_k]
                            self.optimization(u, i, k)
                            item_n = choice(itemList)
                            while self.PositiveSet[user].has_key(item_n) or self.JointSet[user].has_key(item_n)\
                                or self.OkSet[user].has_key(item_n):
                                item_n = choice(itemList)
                            n = self.dao.item[item_n]
                            self.optimization(u,k,n)
                        elif len(jItems)>0:
                            item_j = choice(jItems)
                            j = self.dao.item[item_j]
                            self.optimization(u, i, j)
                            item_n = choice(itemList)
                            while self.PositiveSet[user].has_key(item_n) or self.JointSet[user].has_key(item_n)\
                                or self.OkSet[user].has_key(item_n):
                                item_n = choice(itemList)
                            n = self.dao.item[item_n]
                            self.optimization(u,j,n)

                        else:
                            item_j = choice(itemList)
                            while self.PositiveSet[user].has_key(item_j):
                                item_j = choice(itemList)
                            j = self.dao.item[item_j]
                            self.optimization(u, i, j)

                for friend in self.FNet[user]:
                    f = self.dao.user[friend]
                    err1 = self.FNet[user][friend]-self.P[u].dot(self.M1).dot(self.F[f])
                    self.P[u]+=self.lRate*self.lambda_f*err1*self.M1.dot(self.F[f])
                    self.F[f]+=self.lRate*self.lambda_f * err1 * self.M1.dot(self.P[u])
                    self.M1+=self.lRate*self.lambda_f * err1 * self.P[u].reshape(self.k,1).dot(self.F[f].reshape(1,self.k))
                    #self.loss+=err1**2
                for member in self.MNet[user]:
                    m = self.dao.user[member]
                    err2 = self.MNet[user][member]-self.P[u].dot(self.M2).dot(self.Z[m])
                    self.P[u]+=self.lRate*self.lambda_m*err2*self.M2.dot(self.Z[m])
                    self.Z[m]+=self.lRate*self.lambda_m*err2*self.M2.dot(self.P[u])
                    self.M2 += self.lRate * self.lambda_m*err2*self.P[u].reshape(self.k,1).dot(self.Z[m].reshape(1,self.k))
                    self.loss += err2**2
            # for user in self.dao.user:
            #     if not self.sao.followees.has_key(user):
            #         continueg
            #     for friend in self.sao.followees[user]:
            #         u = self.dao.user[user]
            #         if not self.dao.user.has_key(friend):
            #             continue
            #         f = self.dao.user[friend]
            #         self.P[u] -= 0.1*self.lRate*(self.P[u]-self.P[f])
            self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()#+self.b.dot(self.b)
            iteration += 1
            if self.isConverged(iteration):
                break

    def optimization(self, u, i, j):
        s = sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))
        self.P[u] += self.lRate * (1 - s) * (self.Q[i] - self.Q[j])
        self.Q[i] += self.lRate * (1 - s) * self.P[u]
        self.Q[j] -= self.lRate * (1 - s) * self.P[u]
        self.loss += -log(s)
        self.P[u] -= self.lRate * self.regU * self.P[u]
        self.Q[i] -= self.lRate * self.regI * self.Q[i]
        self.Q[j] -= self.lRate * self.regI * self.Q[j]

    def predict(self,user,item):

        if self.dao.containsUser(user) and self.dao.containsItem(item):
            u = self.dao.getUserId(user)
            i = self.dao.getItemId(item)
            predictRating = sigmoid(self.Q[i].dot(self.P[u]))#+self.b[i])
            return predictRating
        else:
            return sigmoid(self.dao.globalMean)

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            u = self.dao.getUserId(u)
            return self.Q.dot(self.P[u])#+self.b
        else:
            return [self.dao.globalMean] * len(self.dao.item)


