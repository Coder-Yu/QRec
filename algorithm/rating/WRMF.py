from baseclass.IterativeRecommender import IterativeRecommender
import numpy as np
from tool import config
import math
import random

class WRMF(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(WRMF, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(WRMF, self).readConfiguration()
        extraSettings = config.LineConfig(self.config['WRMF'])
        self.alpha = float(extraSettings['-alpha'])

    def buildModel(self):

        Cui = np.mat(np.zeros((self.dao.trainingSize()[0],self.dao.trainingSize()[1])))  # Cui=1+alpha*Rui,alpha=40
        Pui = np.mat(np.zeros((self.dao.trainingSize()[0],self.dao.trainingSize()[1])))  # if Rui>0,Pui = 1;else Pui = 0
        Rui = np.mat(np.zeros((self.dao.trainingSize()[0], self.dao.trainingSize()[1])))  #rating matrix
        userid = [] #userid list
        itemid = [] #itemid list
        for entry in self.dao.trainingData:
            user,item,rating = entry
            if user in userid:
                userindex = userid.index(user)
            else:
                userid.append(user)
                userindex = userid.index(user)

            if item in itemid:
                itemindex = itemid.index(item)
            else:
                itemid.append(item)
                itemindex = itemid.index(item)
            Cui[userindex,itemindex] = 1 + self.alpha * math.log(1+rating/0.01) # * rating
            Pui[userindex,itemindex] = 1
            Rui[userindex,itemindex] = rating

        X = np.mat(np.random.rand(self.dao.trainingSize()[0],self.k))  #user matrix
        Y = np.mat(np.random.rand(self.dao.trainingSize()[1],self.k))   #item matrix
        I_lambda = np.mat(np.eye(self.k,self.k))

        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0

            #update user matrix
            for u in userid:
                index_u = userid.index(u)
                p_u = Pui[index_u, :]
                c_u = Cui[index_u, :]
                CuPu = np.multiply(c_u, p_u).T
                Xu = (np.multiply(Y.T, c_u) * Y + self.regU * I_lambda).I * Y.T * CuPu
                # Xu = (YTY + np.multiply(Y.T, (c_u-1)) * Y + self.regU * I_lambda).I * Y.T * CuPu
                self.P[index_u] = Xu.T
            X = self.P

            #update item matrix
            for i in itemid:
                index_i = itemid.index(i)
                p_i = Pui[:, index_i].T
                c_i = Cui[:, index_i].T
                CiPi = np.multiply(c_i, p_i).T
                Yi = (np.multiply(X.T, c_i) * X + self.regI * I_lambda).I * X.T * CiPi
                # Yi = (XTX + np.multiply(X.T, (c_i-1)) * X+ self.regI * I_lambda).I * X.T * CiPi
                self.Q[index_i] = Yi.T

            error = np.sum(Rui - self.P.dot(self.Q.T))
            self.loss += error ** 2

            Y = self.Q
            #X = self.P

            self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()
            iteration += 1

            if self.isConverged(iteration):
                break






