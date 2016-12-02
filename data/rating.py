import numpy as np
from structure import sparseMatrix,new_sparseMatrix
from tool.config import Config,LineConfig
from tool.qmath import normalize
from evaluation.dataSplit import DataSplit
import os.path
from re import split

class RatingDAO(object):
    'data access control'
    def __init__(self,config):
        self.config = config
        self.ratingConfig = LineConfig(config['ratings.setup'])
        self.user = {} #used to store the order of users
        self.item = {} #used to store the order of items
        self.userMeans = {} #used to store the mean values of users's ratings
        self.itemMeans = {} #used to store the mean values of items's ratings
        self.trainingData = [] #training data
        self.testData = [] #testData
        self.globalMean = 0
        self.timestamp = {}
        self.trainingMatrix = None
        self.validationMatrix = None
        self.testSet_u = {} # used to store the test set by hierarchy user:[item,rating]
        self.testSet_i = {} # used to store the test set by hierarchy item:[user,rating]
        self.rScale = []
        if self.config.contains('evaluation.setup'):
            self.evaluation = LineConfig(config['evaluation.setup'])
            if self.evaluation.contains('-testSet'):
                #specify testSet
                self.__loadDataSet(config['ratings'])
                self.__loadDataSet(self.evaluation['-testSet'],bTest=True)
            elif self.evaluation.contains('-ap'):
                # auto partition
                self.__loadDataSet(config['ratings'])
                self.trainingData,self.testData = DataSplit.\
                    dataSplit(self.trainingData,test_ratio=float(self.evaluation['-ap']))
            self.__generateSet()
        else:
            print 'Evaluation is not well configured!'
            exit(-1)
        self.__computeItemMean()
        self.__computeUserMean()
        self.__globalAverage()

    def __loadDataSet(self,file, bTest=False):
        if not bTest:
            print 'loading training data...'
        else:
            print 'loading test data...'
        with open(file) as f:
            ratings = f.readlines()
        # ignore the headline
        if self.ratingConfig.contains('-header'):
            ratings = ratings[1:]
        # order of the columns
        order = self.ratingConfig['-columns'].strip().split()

        scale = set()
        # find the maximum rating and minimum value

        for lineNo, line in enumerate(ratings):
            items = split(' |,|\t', line.strip())
            if len(order) < 3:
                print 'The rating file is not in a correct format. Error: Line num %d' % lineNo
                exit(-1)
            try:
                rating = items[int(order[2])]
                scale.add(float(rating))
            except ValueError:
                print 'Error! Have you added the option -header to the rating.setup?'
        self.rScale = list(scale)
        self.rScale.sort()

        for lineNo, line in enumerate(ratings):
            items = split(' |,|\t', line.strip())
            if len(order) < 3:
                print 'The rating file is not in a correct format. Error: Line num %d' % lineNo
                exit(-1)
            userId = items[int(order[0])]
            itemId = items[int(order[1])]
            rating = items[int(order[2])]

            if not bTest:
                self.trainingData.append([userId, itemId, float(rating)])
            else:
                self.testData.append([userId, itemId, float(rating)])



    def __generateSet(self):
        triple = []
        for i,entry in enumerate(self.trainingData):
            userId,itemId,rating = entry
            # makes the rating within the range [0, 1].
            rating = normalize(float(rating), self.rScale[-1], self.rScale[0])
            self.trainingData[i][2] = rating
            # order the user
            if not self.user.has_key(userId):
                self.user[userId] = len(self.user)
            # order the item
            if not self.item.has_key(itemId):
                self.item[itemId] = len(self.item)
                # userList.append
            triple.append([self.user[userId], self.item[itemId], rating])
        self.trainingMatrix = new_sparseMatrix.SparseMatrix(triple)

        for entry in self.testData:
            userId, itemId, rating = entry
            # order the user
            if not self.user.has_key(userId):
                self.user[userId] = len(self.user)
            # order the item
            if not self.item.has_key(itemId):
                self.item[itemId] = len(self.item)

            if not self.testSet_u.has_key(userId):
                self.testSet_u[userId] = {}
            self.testSet_u[userId][itemId] = rating
            if not self.testSet_i.has_key(itemId):
                self.testSet_i[itemId] = {}
            self.testSet_i[itemId][userId] = rating




    def __globalAverage(self):
        total = sum(self.userMeans.values())
        if total==0:
            self.globalMean = 0
        else:
            self.globalMean = total/len(self.userMeans)

    def __computeUserMean(self):
        for u in self.user:
            n = self.row(u) > 0
            mean = 0

            if not self.containsUser(u):  # no data about current user in training set
                pass
            else:
                sum = float(self.row(u)[0].sum())
                try:
                    mean =  sum/ n[0].sum()
                except ZeroDivisionError:
                    mean = 0
            self.userMeans[u] = mean

    def __computeItemMean(self):
        for c in self.item:
            n = self.col(c) > 0
            mean = 0
            if not self.containsItem(c):  # no data about current user in training set
                pass
            else:
                sum = float(self.col(c)[0].sum())
                try:
                    mean = sum / n[0].sum()
                except ZeroDivisionError:
                    mean = 0
            self.itemMeans[c] = mean

    def getUserId(self,u):
        if self.user.has_key(u):
            return self.user[u]
        else:
            return -1

    def getItemId(self,i):
        if self.item.has_key(i):
            return self.item[i]
        else:
            return -1

    def trainingSize(self):
        return (self.trainingMatrix.size[0],self.trainingMatrix.size[1],len(self.trainingData))

    def testSize(self):
        return (len(self.testSet_u),len(self.testSet_i),len(self.testData))

    def contains(self,u,i):
        'whether user u rated item i'
        return self.trainingMatrix.contains(self.getUserId(u),self.getItemId(i))

    def containsUser(self,u):
        'whether user is in training set'
        return self.trainingMatrix.matrix_User.has_key(self.getUserId(u))

    def containsItem(self,i):
        'whether item is in training set'
        return self.trainingMatrix.matrix_Item.has_key(self.getItemId(i))

    def userRated(self,u):
        if self.trainingMatrix.matrix_User.has_key(self.getUserId(u)):
            userIndex =  self.trainingMatrix.matrix_User[self.user[u]].keys()
            rating = self.trainingMatrix.matrix_User[self.user[u]].values()
            return (userIndex,rating)
        return ([],[])

    def itemRated(self,i):
        if self.trainingMatrix.matrix_Item.has_key(self.getItemId(i)):
            itemIndex = self.trainingMatrix.matrix_Item[self.item[i]].keys()
            rating = self.trainingMatrix.matrix_Item[self.item[i]].values()
            return (itemIndex,rating)
        return ([],[])

    def row(self,u):
        return self.trainingMatrix.row(self.getUserId(u))

    def col(self,c):
        return self.trainingMatrix.col(self.getItemId(c))

    def rating(self,u,c):
        return self.trainingMatrix.elem(self.getUserId(u),self.getItemId(c))

    def ratingScale(self):
        return (self.rScale[0],self.rScale[1])

    def elemCount(self):
        return self.trainingMatrix.elemCount()
