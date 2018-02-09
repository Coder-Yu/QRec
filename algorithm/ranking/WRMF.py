################# Confidence Frequency Matrix Factorization #################
#                   Weighted Rating Matrix  Factorization                   #
# this algorithm refers to the following paper:
# Yifan Hu et al.Collaborative Filtering for Implicit Feedback Datasets
from baseclass.IterativeRecommender import IterativeRecommender
import math
import numpy as np
from tool import qmath
from random import choice
from tool.qmath import sigmoid
from math import log
from collections import defaultdict
from scipy.sparse import *
from scipy import *

class WRMF(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(WRMF, self).__init__(conf,trainingSet,testSet,fold)

    def initModel(self):
        super(WRMF, self).initModel()
        self.X=self.P*10
        self.Y=self.Q*10
        self.m = self.dao.trainingSize()[0]
        self.n = self.dao.trainingSize()[1]

    def buildModel(self):
        print 'training...'
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            YtY = self.Y.T.dot(self.Y)
            I = np.ones(self.n)
            for user in self.dao.user:
                #C_u = np.ones(self.data.getSize(self.recType))
                H = np.ones(self.n)
                val = []
                pos = []
                P_u = np.zeros(self.n)
                uid = self.dao.user[user]
                for item in self.dao.trainSet_u[user]:
                    iid = self.dao.item[item]
                    r_ui = float(self.dao.trainSet_u[user][item])
                    pos.append(iid)
                    val.append(10*r_ui)
                    H[iid]+=10*r_ui
                    P_u[iid]=1
                    error = (P_u[iid]-self.X[uid].dot(self.Y[iid]))
                    self.loss+=pow(error,2)
                #sparse matrix
                C_u = coo_matrix((val,(pos,pos)),shape=(self.n,self.n))
                A = (YtY+np.dot(self.Y.T,C_u.dot(self.Y))+self.regU*np.eye(self.k))
                self.X[uid] = np.dot(np.linalg.inv(A),(self.Y.T*H).dot(P_u))


            XtX = self.X.T.dot(self.X)
            I = np.ones(self.m)
            for item in self.dao.item:
                P_i = np.zeros(self.m)
                iid = self.dao.item[item]
                H = np.ones(self.m)
                val = []
                pos = []
                for user in self.dao.trainSet_i[item]:
                    uid = self.dao.user[user]
                    r_ui = float(self.dao.trainSet_i[item][user])
                    pos.append(uid)
                    val.append(10*r_ui)
                    H[uid] += 10*r_ui
                    P_i[uid] = 1
                # sparse matrix
                C_i = coo_matrix((val, (pos, pos)),shape=(self.m,self.m))
                A = (XtX+np.dot(self.X.T,C_i.dot(self.X))+self.regU*np.eye(self.k))
                self.Y[iid]=np.dot(np.linalg.inv(A), (self.X.T*H).dot(P_i))

            #self.loss += (self.X * self.X).sum() + (self.Y * self.Y).sum()
            iteration += 1
            print 'iteration:',iteration,'loss:',self.loss
            # if self.isConverged(iteration):
            #     break

    def predictForRanking(self,u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            u = self.dao.getUserId(u)
            return self.Q.dot(self.P[u])+self.dao.globalMean
        else:
            return [self.dao.globalMean] * len(self.dao.item)