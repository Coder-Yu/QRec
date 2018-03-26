#coding:utf-8
from baseclass.IterativeRecommender import IterativeRecommender
import math
import numpy as np
from tool import qmath
from random import choice
from tool.qmath import sigmoid
from math import log
from collections import defaultdict

class ALS(IterativeRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, fold='[1]'):
        super(ALS, self).__init__(conf, trainingSet, testSet, fold)


    def buildModel(self):
        #P是用户矩阵，Q是项目矩阵
        self.lamda = 0.5
        self.alpha = 10
        self.iterationNum = 100
        self.R =np.zeros((self.dao.trainingSize()[0], self.dao.trainingSize()[1]))#评分矩阵初始化：生成m行n列的0矩阵，m是训练集的用户个数，n是项目个数
        self.PUI = np.zeros((self.dao.trainingSize()[0], self.dao.trainingSize()[1]))#pui
        self.C = np.zeros((self.dao.trainingSize()[0], self.dao.trainingSize()[1]))#cui
        for user in self.dao.trainSet_u:#把训练集里的用户、项目、评分——转换为一个评分矩阵，Rij元素是用户i对项目j的评分
            for item in self.dao.trainSet_u[user]:#self.user存的是把训练集里的用户排序了，重新编号，因为可能有一些用户缺失，所以把编号重新排序了，self.item同理。
                self.R[self.dao.user[user]][self.dao.item[item]] =self.dao.trainSet_u[user][item]#trainSet_u（dict）：used to store the train set by hierarchy user:[item,rating]
        for row in range(len(self.R)):#把评分矩阵rui转换为pui和cui
            for column in range(len(self.R[row])):
                self.C[row][column] = 1+log(1+self.R[row][column]/self.alpha)
                if self.R[row][column] !=0:
                    self.PUI[row][column] =1
                else:
                    pass

        for iteration in range(self.iterationNum):
            self.loss = 0
            print 'iteration: '+str(iteration+1)
            YTY = self.Q.T.dot(self.Q)
            for user in self.dao.user:
                '''与一个对角矩阵点乘等于与对角线元素数乘'''
                uid = self.dao.user[user]
                cu = self.C[uid]
                pu = self.PUI[uid]
                for item in self.dao.trainSet_u[user]:
                    iid = self.dao.item[item]
                    error = (pu[iid] - self.P[uid].dot(self.Q[iid]))
                    self.loss += cu[iid] * pow(error, 2)
                I = np.ones(self.dao.trainingSize()[1])
                X = YTY+(self.Q.T*(cu-I)).dot(self.Q)+self.lamda*np.eye(self.k)
                x = (X**-1).dot(self.Q.T)
                self.P[uid] = (x*cu).dot(pu)
                # X = ((self.Q.T * cu).dot(self.Q) + self.lamda * np.eye(self.k)) ** -1
                # self.P[user] = (X.dot(self.Q.T) * cu).dot(pu)


            XTX = self.P.T.dot(self.P)
            for item in self.dao.item:
                iid = self.dao.item[item]
                ci = self.C.T[iid]
                pi = self.PUI.T[iid]
                I = np.ones(self.dao.trainingSize()[0])
                Y = XTX+(self.P.T*(ci-I)).dot(self.P)+self.lamda*np.eye(self.k)
                y = (Y**-1).dot(self.P.T)
                self.Q[iid] = (y*ci).dot(pi)
                # Y = ((self.P.T * ci).dot(self.P) + self.lamda * np.eye(self.k)) ** -1
                # self.Q[item] = (Y.dot(self.P.T) * ci).dot(pi)

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






