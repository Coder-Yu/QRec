from baseclass.IterativeRecommender import IterativeRecommender
import math
import numpy as np
from tool import qmath
from random import choice
from tool.qmath import sigmoid
from math import log
from collections import defaultdict

class WRMF(IterativeRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, fold='[1]'):
        super(WRMF, self).__init__(conf, trainingSet, testSet, fold)

    def initModel(self):
        super(WRMF,self).initModel()
        #read the data
        self.X=self.P
        self.Y=self.Q
    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter: 
            self.loss = 0
            YtY = self.Y.T.dot(self.Y)
            # the user n*n ==> I
            I = np.ones(len(self.dao.item))
            for user in self.dao.user:#self.dao.user:
                C_u = np.ones(len(self.dao.item))
                P_u = np.zeros(len(self.dao.item))
                # uid = self.dao.user[user]
                uid=self.dao.getUserId(user)
                for item in self.dao.trainSet_u[user]:
                    iid =self.dao.getItemId(item)
                    r_ui = self.dao.trainSet_u[user][item]
                    C_u[iid]+=log(1+r_ui/0.01)
                    P_u[iid]=1
                    error = (P_u[iid]-self.X[uid].dot(self.Y[iid]))
                    self.loss+=C_u[iid]*pow(error,2)
                Temp = (YtY+(self.Y.T*(C_u-I)).dot(self.Y)+self.regU*np.eye(self.k))**-1
                self.X[uid] = (Temp.dot(self.Y.T)*C_u).dot(P_u)
            
            XtX = self.X.T.dot(self.X)
            I = np.ones(len(self.dao.user)) 
            for item in self.dao.item:
                C_i = np.ones(len(self.dao.user))
                P_i = np.zeros(len(self.dao.user))
                # iid=self.dao.item[item]
                iid=self.dao.getItemId(item)
                for user in self.dao.trainSet_i[item]:
                    uid = self.dao.getUserId(user)
                    r_ui = self.dao.trainSet_i[item][user]
                    C_i[uid] += log(r_ui/0.01+1)
                    P_i[uid] = 1
                Temp = (XtX+(self.X.T*(C_i-I)).dot(self.X)+self.regU*np.eye(self.k))**-1
                self.Y[iid] = (Temp.dot(self.X.T)*C_i).dot(P_i)
            iteration += 1
            print 'iteration:',iteration
            # if self.isConverged(iteration):
            #     break
    def predict(self, u):
        'invoked to rank all the items for the user'
        u = self.dao.getUserId(u)
        return self.Y.dot(self.X[u])