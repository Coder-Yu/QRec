from base.socialRecommender import SocialRecommender
from math import log
import numpy as np
from util import config
from util.qmath import sigmoid
from random import choice
from collections import defaultdict
class TBPR(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(TBPR, self).__init__(conf,trainingSet,testSet,relation,fold)

    def readConfiguration(self):
        super(TBPR, self).readConfiguration()
        options = config.OptionConf(self.config['TBPR'])
        self.regT = float(options['-regT'])

    def initModel(self):
        super(TBPR, self).initModel()
        self.strength = defaultdict(dict)
        self.weakTies = defaultdict(dict)
        self.strongTies = defaultdict(dict)
        self.weights = []
        for u1 in self.social.user:
            N_u1 = list(self.social.getFollowees(u1).keys())
            for u2 in self.social.getFollowees(u1):
                if u1==u2:
                    continue
                N_u2 = list(self.social.getFollowees(u2).keys())
                s = len(set(N_u1).intersection(set(N_u2)))/(len(set(N_u1).union(set(N_u2)))+0.0)
                self.strength[u1][u2]=s
                self.weights.append(s)
        self.weights.sort()
        self.weights = np.array(self.weights)
        self.theta = np.median(self.weights)
        for u1 in self.strength:
            for u2 in self.strength[u1]:
                if self.strength[u1][u2]>self.theta:
                    self.strongTies[u1][u2]=self.strength[u1][u2]
                else:
                    self.weakTies[u1][u2]=self.strength[u1][u2]
        self.t_s = self.weights[len(self.weights)//2+1:].sum()/(len(self.weights[len(self.weights)//2+1:])+0.0)
        self.t_w = self.weights[0:len(self.weights)//2].sum()/(len(self.weights[0:len(self.weights)//2])+0.0)

    def optimization(self,u,i,j):
        s = sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))
        self.P[u] += self.lRate * (1 - s) * (self.Q[i] - self.Q[j])
        self.Q[i] += self.lRate * (1 - s) * self.P[u]
        self.Q[j] -= self.lRate * (1 - s) * self.P[u]
        self.loss += -log(s)
        self.P[u] -= self.lRate * self.regU * self.P[u]
        self.Q[i] -= self.lRate * self.regI * self.Q[i]
        self.Q[j] -= self.lRate * self.regI * self.Q[j]

    def optimization_theta(self,u,i,j):
        # s = sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))
        # self.P[u] += self.lRate * (1 - s) * (self.Q[i] - self.Q[j])
        # self.Q[i] += self.lRate * (1 - s) * self.P[u]
        # self.Q[j] -= self.lRate * (1 - s) * self.P[u]
        # self.loss += -log(s)
        # self.P[u] -= self.lRate * self.regU * self.P[u]
        # self.Q[i] -= self.lRate * self.regI * self.Q[i]
        # self.Q[j] -= self.lRate * self.regI * self.Q[j]
        s = sigmoid((self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))/(1+1/self.g_theta))
        self.P[u] += self.lRate * 1/(1+1/self.g_theta)*(1 - s) * (self.Q[i] - self.Q[j])
        self.Q[i] += self.lRate * 1/(1+1/self.g_theta)*(1 - s) * self.P[u]
        self.Q[j] -= self.lRate * 1/(1+1/self.g_theta)*(1 - s) * self.P[u]
        self.loss += -log(s)
        self.P[u] -= self.lRate * self.regU * self.P[u]
        self.Q[i] -= self.lRate * self.regI * self.Q[i]
        self.Q[j] -= self.lRate * self.regI * self.Q[j]
        self.theta_derivative += self.regT * self.theta + ((1 - s)*(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))*(self.t_w+self.t_s-2*self.theta))/(self.g_theta+1)**2
        self.theta_count+=1

    def trainModel(self):
        self.positiveSet = defaultdict(dict)
        for user in self.data.user:
            for item in self.data.trainSet_u[user]:
                if self.data.trainSet_u[user][item] >= 1:
                    self.positiveSet[user][item] = 1

        print('Training...')
        epoch = 0
        while epoch < self.maxEpoch:
            self.theta_derivative=0
            self.theta_count = 0
            if self.theta>self.weights.max():
                self.theta=self.weights.max()-0.01
            if self.theta<self.weights.min():
                self.theta=self.weights.min()+0.01
            try:
                self.t_s = sum([item for item in self.weights if item >= self.theta])/len([item for item in self.weights if item >= self.theta])
                self.t_w = sum([item for item in self.weights if item <= self.theta])/len([item for item in self.weights if item <= self.theta])
            except ZeroDivisionError:
                self.t_w = 0.01
                self.theta=0.02
                #pass
            # if self.theta==0:
            #     self.theta=0.02
            self.g_theta = (self.t_s-self.theta)*(self.theta-self.t_w)
            print('Theta:',self.theta)
            print('g_theta:',self.g_theta)
            print('Preparing item sets...')

            self.jointSet = defaultdict(dict)
            self.strongSet = defaultdict(dict)
            self.weakSet = defaultdict(dict)

            for u1 in self.social.user:
                if u1 in self.data.user:
                    for u2 in self.strongTies[u1]:
                        for item in self.data.trainSet_u[u2]:
                            if self.data.trainSet_u[u2][item] >= 1 and item not in self.positiveSet[u1]:
                                self.strongSet[u1][item]=1

                    for u2 in self.weakTies[u1]:
                        for item in self.data.trainSet_u[u2]:
                            if self.data.trainSet_u[u2][item] >= 1 and item not in self.positiveSet[u1]:
                                self.weakSet[u1][item]=1

            for u1 in self.social.user:
                if u1 in self.data.user:
                    self.jointSet[u1] = dict.fromkeys(set(self.strongSet[u1].keys()).intersection(set(self.weakSet[u1].keys())),1)

            for u1 in self.jointSet:
                for item in self.jointSet[u1]:
                    del self.strongSet[u1][item]
                    del self.weakSet[u1][item]
                    if len(self.strongSet[u1])==0:
                        del self.strongSet[u1]
                    if len(self.weakSet[u1])==0:
                        del self.weakSet[u1]

            print('Computing...')
            self.loss = 0
            itemList = list(self.data.item.keys())
            for user in self.positiveSet:
                #print user
                u = self.data.user[user]
                jItems = list(self.jointSet[user].keys())
                wItems = list(self.weakSet[user].keys())
                sItems = list(self.strongSet[user].keys())
                for item in self.positiveSet[user]:
                    i = self.data.item[item]
                    selectedItems = [i]
                    if len(jItems)>0:
                        item_j = choice(jItems)
                        j = self.data.item[item_j]
                        selectedItems.append(j)
                    if len(wItems) > 0:
                        item_w = choice(wItems)
                        w = self.data.item[item_w]
                        selectedItems.append(w)
                    if len(sItems) > 0:
                        item_s = choice(sItems)
                        s = self.data.item[item_s]
                        selectedItems.append(s)
                    item_k = choice(itemList)
                    while item_k in self.positiveSet[user]:
                        item_k = choice(itemList)
                    k = self.data.item[item_k]
                    selectedItems.append(k)
                    # optimization
                    for ind, item in enumerate(selectedItems[:-1]):
                        self.optimization(u, item, selectedItems[ind + 1])
                self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()
            if self.theta_count>0:
                self.theta -= self.lRate*self.theta_derivative/self.theta_count
                self.weakTies = defaultdict(dict)
                self.strongTies = defaultdict(dict)
                for u1 in self.strength:
                    for u2 in self.strength[u1]:
                        if self.strength[u1][u2] > self.theta:
                            self.strongTies[u1][u2] = self.strength[u1][u2]
                        else:
                            self.weakTies[u1][u2] = self.strength[u1][u2]
            epoch += 1
            if self.isConverged(epoch):
                break

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.Q.dot(self.P[u])
        else:
            return [self.data.globalMean] * self.num_items

