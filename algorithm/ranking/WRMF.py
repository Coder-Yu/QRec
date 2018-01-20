from baseclass.IterativeRecommender import IterativeRecommender
import numpy as np
from collections import defaultdict
import math
''' first trial:failed version,very slow and full of mistakes
class WRMF(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(WRMF, self).__init__(conf,trainingSet,testSet,fold)

    def initModel(self):
        super(WRMF, self).initModel()

    def buildModel(self):
        #record Pu
        self.PreferenceSet1 = defaultdict(dict)<----saved data in dictionary,resulting in a very slow speed,user matrices as much as possible
        #record Pi
        self.PreferenceSet2 = defaultdict(dict)
        #record Cu
        self.ConfidenceSet1 = defaultdict(dict)
        #record Ci
        self.ConfidenceSet2 = defaultdict(dict)
        iteration = 0
        factors=10
        RegRate=0.1
        I=np.eye(factors)
        userNum=len(self.dao.trainSet_u)
        itemNum=len(self.dao.trainSet_i)
        # initialize user-item set
        for user in self.dao.user:
            for item in self.dao.item:
                rating = 0
                if self.dao.contains(user, item):
                    rating = self.dao.trainSet_u[user][item]
                if rating >= (1.0 / 2.0):
                    self.PreferenceSet1[user][item] = 1
                    self.PreferenceSet2[item][user] = 1
                else:
                    self.PreferenceSet1[user][item] = 0
                    self.PreferenceSet2[item][user] = 0
                self.ConfidenceSet1[user][item] = 1 + 1 * math.log(1 + rating / 0.01)
                self.ConfidenceSet2[item][user] = 1 + 1 * math.log(1 + rating / 0.01)

        while iteration < self.maxIter:
            self.loss = 0

            #update parameters once
            #update parameter alternatively,update Xu when k is odd,otherwise update Yu
                #calculate Y transpose multiply C to the power of U
            matrixY=np.zeros((factors,itemNum))
            for user in self.dao.user:
                uid = self.dao.getUserId(user)
                # calculate preference of user
                Puser=np.zeros((userNum,itemNum))
                Puser[uid] = np.array(self.PreferenceSet1[user].values())
                for item in self.dao.item:
                    iid=self.dao.getItemId(item)
                    matrixY[:, iid]=(np.transpose(self.Q))[:, iid]*self.ConfidenceSet1[user][item]
                self.P[uid]=((np.linalg.inv((matrixY.dot(self.Q)+RegRate*I))).dot(matrixY)).dot(Puser[uid])

                # calculate X transpose multiply C to the power of U
            matrixX = np.zeros((factors,userNum))
            for item in self.dao.item:
                iid=self.dao.getItemId(item)
                # calculate preference of item(all user to item i)
                Pitem = np.array(self.PreferenceSet2[item].values())
                for user in self.dao.user:
                    uid = self.dao.getUserId(user)
                    matrixX[:, uid] = (np.transpose(self.P))[:, uid] * self.ConfidenceSet2[item][user]
                self.Q[iid] = ((np.transpose((matrixX.dot(self.P) + RegRate * I))).dot(matrixX)).dot(Pitem)

            for item in self.dao.item:
                iid = self.dao.item[item]
                for user in self.dao.user:
                    uid = self.dao.user[user]
                    #self.loss=self.loss+self.ConfidenceSet2[item][user]*((self.PreferenceSet2[item][user]-(np.transpose(self.P[uid])).dot(self.Q[iid]))**2)
            #iteration += 1
            #if self.isConverged(iteration):
                #break
'''
class WRMF(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(WRMF, self).__init__(conf,trainingSet,testSet,fold)

    def initModel(self):
        super(WRMF, self).initModel()

    def buildModel(self):
        iteration = 0
        #generate f*f factor matrix
        I = np.eye(20)
        #initialize matrix p,r,and c
        userNum = len(self.dao.user)
        itemNum = len(self.dao.item)
        preference=np.zeros((userNum,itemNum))
        rating=np.zeros((userNum,itemNum))
        confidence=np.ones((userNum,itemNum))
        array1=np.ones(itemNum)
        array2=np.ones(userNum)
        for user in self.dao.user:
            for item in self.dao.trainSet_u[user]:
                uid=self.dao.getUserId(user)
                iid=self.dao.getItemId(item)
                rating[uid][iid]=self.dao.trainSet_u[user][item]
                #rated pairs u-i,set its preference=1
                preference[uid][iid]=1
                confidence[uid][iid]=1 + 1 * math.log(1 + rating[uid][iid]/0.01)
        print'start predicting:'
        #do the iteration
        while iteration<self.maxIter:
            self.loss=0
            #update P[u]
            YtY=self.Q.T.dot(self.Q)
            #calculate Yt(Cu-I)
            YtCuI=self.Q.T
            for user in self.dao.user:
                uid = self.dao.getUserId(user)
                #most important step matrix*row vector means multiplying matrix's each row by the corresponding scalar in vector,the same goes for column vector
                #exmaple([[2,3,3],[2,1,3]])*([1,2,2])=([[2,6,6],[2,2,6]])
                #Previously I wrote another for item in self.dao.trainSet_u
                #YtCui[:,iid]=self.Q.T[:,iid]*(confidence[iid][uid]-1) it was much slower and it was wrong
                YtCuI=self.Q.T*(confidence[uid]-array1)
                uid = self.dao.getUserId(user)
                self.P[uid]=(np.linalg.inv(YtY+YtCuI.dot(self.Q)+I).dot(YtCuI+self.Q.T)).dot(preference[uid])

            #update Q[i]
            XtCuI=self.P.T
            XtX=self.P.T.dot(self.P)
            #calculate Xt(Cu-I)
            for item in self.dao.item:
                iid = self.dao.getItemId(item)
                XtCuI=self.P.T*(confidence[:,iid]-array2)
                iid = self.dao.getItemId(item)
                self.Q[iid]=(np.linalg.inv(XtX+XtCuI.dot(self.P)+I)).dot(XtCuI+self.P.T).dot(preference[:,iid])
                #without this for loop running time will be cut to 20s,with this loop running time is 132s
                for user in self.dao.user:
                    uid=self.dao.getUserId(user)
                    self.loss=self.loss+confidence[uid][iid]*((preference[uid][iid]-(self.P[uid].T).dot(self.Q[iid]))**2)
            iteration += 1
            if self.isConverged(iteration):
                break