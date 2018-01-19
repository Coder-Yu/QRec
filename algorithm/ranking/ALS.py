#coding:utf8
from baseclass.IterativeRecommender import IterativeRecommender
from tool import config
import numpy as np
from tool.qmath import sigmoid
from tool.qmath import denormalize
from collections import defaultdict
from math import log

class ALS(IterativeRecommender):
    # ALS：Alternating Least Square for Weighted Regularized Matrix Factorization
    # Hu et al., Collaborative filtering for implicit feedback datasets, ICDM 2008

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(ALS, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(ALS, self).readConfiguration()

    def initModel(self):
        super(ALS, self).initModel()

    def buildModel(self):
        print 'run the MF_ALS algorithm'   
       
        print 'training...'        
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            I = np.ones(len(self.dao.item))
            for user in self.dao.user:
                C_u = np.ones(len(self.dao.item))
                P_u = np.zeros(len(self.dao.item))
                uid = self.dao.user[user]
                for item in self.dao.trainSet_u[user]:
                    iid = self.dao.getItemId(item)
                    r_ui = denormalize(self.dao.trainSet_u[user][item],self.dao.rScale[-1], self.dao.rScale[0])
                    C_u[iid] += log(1+r_ui)
                    P_u[iid] = 1
                    error = (P_u[iid]-self.P[uid].dot(self.Q[iid]))
                    self.loss += C_u[iid]*error**2
                    
                Temp = (self.Q.T.dot(self.Q)+(self.Q.T*(C_u-I)).dot(self.Q) + self.regU*np.eye(self.k))**-1
                self.P[uid] = (Temp.dot(self.Q.T)*C_u).dot(P_u)
            
            I = np.ones(len(self.dao.user)) 
            for item in self.dao.item:
                C_i = np.ones(len(self.dao.user))
                P_i = np.zeros(len(self.dao.user))
                iid = self.dao.item[item]
                for user in self.dao.trainSet_i[item]:
                    uid = self.dao.getUserId(user)
                    r_ui = denormalize(self.dao.trainSet_i[item][user],self.dao.rScale[-1], self.dao.rScale[0])
                    C_i[uid] += log(r_ui+1)
                    P_i[uid] = 1
                Temp = (self.P.T.dot(self.P)+(self.P.T*(C_i-I)).dot(self.P) + self.regU*np.eye(self.k))**-1
                self.Q[iid] = (Temp.dot(self.P.T)*C_i).dot(P_i)
            
            # self.loss += (self.P * self.P).sum() + (self.Q * self.Q).sum()
            
            iteration += 1
            if self.isConverged(iteration):
                break


    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        u = self.dao.getUserId(u)
        return self.Q.dot(self.P[u])


