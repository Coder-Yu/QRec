import numpy as np
from structure import sparseMatrix,new_sparseMatrix
from tool.config import Config,LineConfig
from tool.qmath import normalize
from evaluation.dataSplit import DataSplit
import os.path
from re import split
from collections import defaultdict
class RatingDAO(object):
    'data access control'
    def __init__(self,config,trainingSet, testSet):
        self.config = config
        self.ratingConfig = LineConfig(config['ratings.setup'])
        self.user = {} #used to store the order of users in the training set
        self.item = {} #used to store the order of items in the training set
        self.id2user = {}
        self.id2item = {}
        self.all_Item = {}
        self.all_User = {}
        self.userMeans = {} #used to store the mean values of users's ratings
        self.itemMeans = {} #used to store the mean values of items's ratings
        self.globalMean = 0
        self.timestamp = {}
        self.trainSet_u = defaultdict(dict)
        self.trainSet_i = defaultdict(dict)
        self.testSet_u = defaultdict(dict) # used to store the test set by hierarchy user:[item,rating]
        self.testSet_i = defaultdict(dict) # used to store the test set by hierarchy item:[user,rating]
        self.rScale = []

        self.trainingData = trainingSet[:]
        self.testData = testSet[:]

        self.__generateSet()

        self.__computeItemMean()
        self.__computeUserMean()
        self.__globalAverage()



    def __generateSet(self):
        triple = []
        scale = set()
        # find the maximum rating and minimum value

        for i,entry in enumerate(self.trainingData):
            userName,itemName,rating = entry
            # makes the rating within the range [0, 1].
            #rating = normalize(float(rating), self.rScale[-1], self.rScale[0])
            #self.trainingData[i][2] = rating
            # order the user
            if not self.user.has_key(userName):
                self.user[userName] = len(self.user)
                self.id2user[self.user[userName]] = userName
            # order the item
            if not self.item.has_key(itemName):
                self.item[itemName] = len(self.item)
                self.id2item[self.item[itemName]] = itemName
                # userList.append
            self.trainSet_u[userName][itemName] = rating
            self.trainSet_i[itemName][userName] = rating
            scale.add(float(rating))
        self.rScale = list(scale)
        self.rScale.sort()

        self.all_User.update(self.user)
        self.all_Item.update(self.item)
        for entry in self.testData:
            userName, itemName, rating = entry
            # order the user
            if not self.user.has_key(userName):
                self.all_User[userName] = len(self.all_User)
            # order the item
            if not self.item.has_key(itemName):
                self.all_Item[itemName] = len(self.all_Item)

            self.testSet_u[userName][itemName] = rating
            self.testSet_i[itemName][userName] = rating




    def __globalAverage(self):
        total = sum(self.userMeans.values())
        if total==0:
            self.globalMean = 0
        else:
            self.globalMean = total/len(self.userMeans)

    def __computeUserMean(self):
        for u in self.user:
            # n = self.row(u) > 0
            # mean = 0
            #
            # if not self.containsUser(u):  # no data about current user in training set
            #     pass
            # else:
            #     sum = float(self.row(u)[0].sum())
            #     try:
            #         mean =  sum/ n[0].sum()
            #     except ZeroDivisionError:
            #         mean = 0
            self.userMeans[u] = sum(self.trainSet_u[u].values())/float(len(self.trainSet_u[u]))

    def __computeItemMean(self):
        for c in self.item:
            self.itemMeans[c] = sum(self.trainSet_i[c].values()) / float(len(self.trainSet_i[c]))

    def getUserId(self,u):
        if self.user.has_key(u):
            return self.user[u]

    def getItemId(self,i):
        if self.item.has_key(i):
            return self.item[i]

    def trainingSize(self):
        return (len(self.user),len(self.item),len(self.trainingData))

    def testSize(self):
        return (len(self.testSet_u),len(self.testSet_i),len(self.testData))

    def contains(self,u,i):
        'whether user u rated item i'
        if self.user.has_key(u) and self.trainSet_u[u].has_key(i):
            return True
        else:
            return False


    def containsUser(self,u):
        'whether user is in training set'
        if self.user.has_key(u):
            return True
        else:
            return False

    def containsItem(self,i):
        'whether item is in training set'
        if self.item.has_key(i):
            return True
        else:
            return False

    def userRated(self,u):
        return self.trainSet_u[u].keys(),self.trainSet_u[u].values()

    def itemRated(self,i):
        return self.trainSet_i[i].keys(),self.trainSet_i[i].values()

    def row(self,u):
        k,v = self.userRated(u)
        vec = np.zeros(len(self.item))
        #print vec
        for pair in zip(k,v):
            iid = self.item[pair[0]]
            vec[iid]=pair[1]
        return vec

    def col(self,i):
        k,v = self.itemRated(i)
        vec = np.zeros(len(self.user))
        #print vec
        for pair in zip(k,v):
            uid = self.user[pair[0]]
            vec[uid]=pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.user),len(self.item)))
        for u in self.user:
            k, v = self.userRated(u)
            vec = np.zeros(len(self.item))
            # print vec
            for pair in zip(k, v):
                iid = self.item[pair[0]]
                vec[iid] = pair[1]
            m[self.user[u]]=vec
        return m
    # def row(self,u):
    #     return self.trainingMatrix.row(self.getUserId(u))
    #
    # def col(self,c):
    #     return self.trainingMatrix.col(self.getItemId(c))

    def sRow(self,u):
        return self.trainSet_u[u]

    def sCol(self,c):
        return self.trainSet_i[c]

    def rating(self,u,c):
        if self.contains(u,c):
            return self.trainSet_u[u][c]
        return -1

    def ratingScale(self):
        return (self.rScale[0],self.rScale[1])

    def elemCount(self):
        return len(self.trainingData)
