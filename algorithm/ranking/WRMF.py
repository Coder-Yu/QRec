#coding:utf8
from baseclass.IterativeRecommender import IterativeRecommender
import math
import numpy as np
from tool import qmath
from random import choice
from tool.qmath import sigmoid
from math import log
from collections import defaultdict
class WRMF(IterativeRecommender):


    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(WRMF, self).__init__(conf,trainingSet,testSet,fold)


    def initModel(self):
        super(WRMF, self).initModel()

    def buildModel(self):
        #print self.dao.id2user
        #print self.dao.trainSet_u
        alpha = 1
        lamda = 0.5
        
        
        
        self.Rui = np.zeros((self.dao.trainingSize()[0],self.dao.trainingSize()[1]))
        self.Pui = np.zeros((self.dao.trainingSize()[0],self.dao.trainingSize()[1]))
        self.Cui = np.zeros((self.dao.trainingSize()[0],self.dao.trainingSize()[1]))
        
        for user in self.dao.trainSet_u:
            for item in self.dao.trainSet_u[user]:
                self.Rui[self.dao.user[user]][self.dao.item[item]] = self.dao.trainSet_u[user][item]
                self.Pui[self.dao.user[user]][self.dao.item[item]] = 1
                self.Cui[self.dao.user[user]][self.dao.item[item]] = alpha*log(1+self.dao.trainSet_u[user][item]/0.01)+1
        '''
        
        rui = []
        pui = []
        
        for user in self.dao.user:
            ru = []
            pu = []
            for item in self.dao.item:
                if self.dao.trainSet_u.has_key(user):
                    if self.dao.trainSet_u[user].has_key(item):
                        ru.append(self.dao.trainSet_u[user][item])
                        if self.dao.trainSet_u[user][item] > 0:
                            pu.append(1)
                        else:
                            pu.append(0)
                    else:
                        ru.append(0)
                        pu.append(0)
                else:
                    ru.append(0)
                    pu.append(0)
            rui.append(ru)
            pui.append(pu)
            
        self.Rui = np.array(rui)
        self.Pui = np.array(pui)
        self.Cui = alpha*self.Rui+1
        '''
        print 'Start...'
        iteration = 0
        while iteration < self.maxIter:
            
            '''
            i = 0
            u = 0
            for user in self.dao.user:
                Y = self.Q
                YT = self.Q.T
                Cu = Cui[u]
                Pu = Pui[u]
                #print np.linalg.inv(YT.dot(Y) + (YT.dot(np.diag(Cu)-np.eye(self.dao.trainingSize()[1]))).dot(Y) + lamda*np.eye(self.k)).dot(YT).dot(np.diag(Cu)).dot(Pu)
                #YTCUY_LamdaI = YT.dot(Y) + (YT.dot(np.diag(Cu)-np.eye(self.dao.trainingSize()[1]))).dot(Y) + lamda*np.eye(self.k)
                #Yi = np.linalg.inv(YTCUY_LamdaI).dot(YT).dot(np.diag(Cu)).dot(Pu)
                YTCUY_LamdaI = YT.dot(Y) + (YT*(np.diag(Cu)-np.eye(self.dao.trainingSize()[1]))).dot(Y) + lamda*np.eye(self.k)
                Yi = np.linalg.inv(YTCUY_LamdaI).dot(YT).dot(np.diag(Cu)).dot(Pu)
                self.P[u] = Yi
                u += 1
            
            for item in self.dao.item:
                X = self.P
                XT = self.P.T
                Ci = Cui[:,i]
                Pi = Pui[:,i]
                #print np.linalg.inv(YT.dot(Y) + (YT.dot(np.diag(Cu)-np.eye(self.dao.trainingSize()[1]))).dot(Y) + lamda*np.eye(self.k)).dot(YT).dot(np.diag(Cu)).dot(Pu)
                XTCIX_LamdaI = XT.dot(X) + (XT*(np.diag(Ci)-np.eye(self.dao.trainingSize()[0]))).dot(X) + lamda*np.eye(self.k)
                Xu = np.linalg.inv(XTCIX_LamdaI).dot(XT).dot(np.diag(Ci)).dot(Pi)
                self.Q[i] = Xu
                i += 1
            '''
            
            I = np.ones(self.dao.trainingSize()[1])
            for u in range(len(self.dao.user)):
                Y = self.Q
                YT = self.Q.T
                Cu = self.Cui[u]
                Pu = self.Pui[u]
                
                error = (Pu-self.P[u].dot(YT))
                self.loss+=sum(Cu*pow(error,2))
                #print np.linalg.inv(YT.dot(Y) + (YT.dot(np.diag(Cu)-np.eye(self.dao.trainingSize()[1]))).dot(Y) + lamda*np.eye(self.k)).dot(YT).dot(np.diag(Cu)).dot(Pu)
                YTCUY_LamdaI = YT.dot(Y) + (YT*(Cu-I)).dot(Y) + lamda*np.eye(self.k)
                Yi = (np.linalg.inv(YTCUY_LamdaI).dot(YT)*Cu).dot(Pu)

                self.P[u] = Yi
            
            I = np.ones(self.dao.trainingSize()[0])
            for i in range(len(self.dao.item)):
                X = self.P
                XT = self.P.T
                Ci = self.Cui[:,i]
                Pi = self.Pui[:,i]
                #print np.linalg.inv(YT.dot(Y) + (YT.dot(np.diag(Cu)-np.eye(self.dao.trainingSize()[1]))).dot(Y) + lamda*np.eye(self.k)).dot(YT).dot(np.diag(Cu)).dot(Pu)
                XTCIX_LamdaI = XT.dot(X) + (XT*(Ci-I)).dot(X) + lamda*np.eye(self.k)
                Xu = (np.linalg.inv(XTCIX_LamdaI).dot(XT)*Ci).dot(Pi)
                self.Q[i] = Xu
            
            
            '''
            I = np.ones(self.dao.trainingSize()[1])
            for user in self.dao.trainSet_u):
                Y = self.Q
                YT = self.Q.T
                C_u = np.ones(self.dao.trainingSize()[1])
                P_u = np.zeros(self.dao.trainingSize()[1])
                for item in self.dao.trainSet_u[user]:
                    
                #print np.linalg.inv(YT.dot(Y) + (YT.dot(np.diag(Cu)-np.eye(self.dao.trainingSize()[1]))).dot(Y) + lamda*np.eye(self.k)).dot(YT).dot(np.diag(Cu)).dot(Pu)
                YTCUY_LamdaI = YT.dot(Y) + (YT*(Cu-I)).dot(Y) + lamda*np.eye(self.k)
                Yi = (np.linalg.inv(YTCUY_LamdaI).dot(YT)*Cu).dot(Pu)
                self.P[u] = Yi
            
            I = np.ones(self.dao.trainingSize()[0])
            for i in range(len(self.dao.item)):
                X = self.P
                XT = self.P.T
                Ci = Cui[:,i]
                Pi = Pui[:,i]
                #print np.linalg.inv(YT.dot(Y) + (YT.dot(np.diag(Cu)-np.eye(self.dao.trainingSize()[1]))).dot(Y) + lamda*np.eye(self.k)).dot(YT).dot(np.diag(Cu)).dot(Pu)
                XTCIX_LamdaI = XT.dot(X) + (XT*(Ci-I)).dot(X) + lamda*np.eye(self.k)
                Xu = (np.linalg.inv(XTCIX_LamdaI).dot(XT)*Ci).dot(Pi)
                self.Q[i] = Xu
                '''
            iteration += 1
            print 'iteration:',iteration
            
            if self.isConverged(iteration):
                break
            
   

