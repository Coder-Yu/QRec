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
        self.user = {} #map user names to identifiers (id)
        self.item = {} #map item names to identifiers (id)
        self.id2user = {}
        self.id2item = {}
        self.userMeans = {} #Store the mean values of users's ratings
        self.itemMeans = {} #Store the mean values of items's ratings
        self.globalMean = 0
        self.trainSet_u = defaultdict(dict)
        self.trainSet_i = defaultdict(dict)
        self.testSet_u = defaultdict(dict) # Store the test set in the form of [user][item]=rating
        self.testSet_i = defaultdict(dict) # Store the test set in the form of [item][user]=rating]
        self.rScale = [] #rating scale

        self.trainingData = trainingSet[:]
        self.testData = testSet[:]

        self.__generateSet()
        self.__computeItemMean()
        self.__computeUserMean()
        self.__globalAverage()


    def __generateSet(self):
        scale = set()
        # find the maximum rating and minimum value

        for i,entry in enumerate(self.trainingData):
            userName,itemName,rating = entry
            # makes the rating within the range [0, 1].
            #rating = normalize(float(rating), self.rScale[-1], self.rScale[0])
            #self.trainingData[i][2] = rating
            # order the user
            if userName not in self.user:
                self.user[userName] = len(self.user)
                self.id2user[self.user[userName]] = userName
            # order the item
            if itemName not in self.item:
                self.item[itemName] = len(self.item)
                self.id2item[self.item[itemName]] = itemName
                # userList.append
            self.trainSet_u[userName][itemName] = rating
            self.trainSet_i[itemName][userName] = rating
            scale.add(float(rating))
        self.rScale = list(scale)
        self.rScale.sort()

        for entry in self.testData:
            userName, itemName, rating = entry
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
            self.userMeans[u] = sum(self.trainSet_u[u].values())/float(len(self.trainSet_u[u]))

    def __computeItemMean(self):
        for c in self.item:
            self.itemMeans[c] = sum(self.trainSet_i[c].values()) / float(len(self.trainSet_i[c]))

    def getUserId(self,u):
        if u in self.user:
            return self.user[u]

    def getItemId(self,i):
        if i in self.item:
            return self.item[i]

    def trainingSize(self):
        return (len(self.user),len(self.item),len(self.trainingData))

    def testSize(self):
        return (len(self.testSet_u),len(self.testSet_i),len(self.testData))

    def contains(self,u,i):
        'whether user u rated item i'
        if u in self.user and i in self.trainSet_u[u]:
            return True
        else:
            return False

    def containsUser(self,u):
        'whether user is in training set'
        if u in self.user:
            return True
        else:
            return False

    def containsItem(self,i):
        'whether item is in training set'
        if i in self.item:
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
