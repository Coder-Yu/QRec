import numpy as np
from structure import sparseMatrix,new_sparseMatrix
from tool.config import Config,LineConfig
from tool.qmath import normalize
import os.path
from re import split
#from sklearn.cross_validation import train_test_split
class ratingDAO(object):
    'data access control'
    def __init__(self,config):
        self.config = config
        self.ratingConfig = LineConfig(config['ratings.setup'])
        self.evaluation = LineConfig(config['evaluation.setup'])
        self.user = {} #used to store the order of users
        self.item = {} #used to store the order of items
        self.userMeans = {} #used to store the mean values of users's ratings
        self.itemMeans = {} #used to store the mean values of items's ratings
        self.globalMean = 0
        self.timestamp = {}
        self.ratingMatrix = None
        self.trainingMatrix = None
        self.validationMatrix = None
        self.testSet_u = None # used to store the test set by hierarchy user:[item,rating]
        self.testSet_i = None # used to store the test set by hierarchy item:[user,rating]
        self.rScale = [-9999999,999999]
        if self.evaluation.contains('-testSet'):
            #specify testSet
            self.trainingMatrix = self.__loadRatings(config['ratings'])
            self.testSet_u,self.testSet_i = self.__loadRatings(self.evaluation['-testSet'],True)
        else: #cross validation and leave-one-out
            self.ratingMatrix = self.__loadRatings(config['ratings'])
        self.__computeItemMean()
        self.__computeUserMean()
        self.__globalAverage()



    def __loadRatings(self,file,bTest=False):
        with open(file) as f:
            ratings = f.readlines()
        #ignore the headline
        if self.ratingConfig.contains('-header'):
            ratings = ratings[1:]
        #order of the columns
        order = self.ratingConfig['-columns'].strip().split()
        #split data
        userList= []
        u_i_r = {}
        i_u_r = {}
        triple = []
        for line in ratings:
            items = split(' |,|\t',line.strip())
            userId =  items[int(order[0])]
            itemId =  items[int(order[1])]
            rating =  items[int(order[2])]
            if float(rating) > self.rScale[0]:
                self.rScale[0] = float(rating)
            if float(rating) < self.rScale[1]:
                self.rScale[1] = float(rating)
            #order the user
            if not self.user.has_key(userId):
                self.user[userId] = len(self.user)
            #order the item
            if not self.item.has_key(itemId):
                self.item[itemId] = len(self.item)
            if not u_i_r.has_key(userId):
                u_i_r[userId] = []
                userList.append(userId)
            u_i_r[userId].append([itemId,float(rating)])
            if not i_u_r.has_key(itemId):
                i_u_r[itemId] = []
            i_u_r[itemId].append([userId,float(rating)])
            triple.append([self.user[userId],self.item[itemId],float(rating)])

        if not bTest:
            #contruct the sparse matrix
            # data=[]
            # indices=[]
            # indptr=[]
            # offset = 0
            # for uid in userList:
            #     uRating = [r[1] for r in u_i_r[uid]]
            #     uColunms = [self.item[r[0]] for r in u_i_r[uid]]
            #     data += uRating
            #     indices += uColunms
            #     indptr .append(offset)
            #     offset += len(uRating)
            # indptr.append(offset)
            # return sparseMatrix.SparseMatrix(data, indices, indptr)
            return new_sparseMatrix.SparseMatrix(triple,(len(self.user),len(self.item)))
        else:
            # return testSet
            return u_i_r,i_u_r

    def __globalAverage(self):
        self.globalMean = sum(self.userMeans.values())/len(self.userMeans)

    def __computeUserMean(self):
        for u in self.user:
            n = self.row(u) > 0
            mean = 0
            if n[0].sum() == 0:  # no data about current user in training set
                pass
            else:
                mean = float(self.row(u)[0].sum() / n[0].sum())
            self.userMeans[u] = mean

    def __computeItemMean(self):
        for c in self.item:
            n = self.col(c) > 0
            mean = 0
            if n[0].sum() == 0:  # no data about current user in training set
                pass
            else:
                mean = float(self.col(c)[0].sum() / n[0].sum())
            self.itemMeans[c] = mean

    def contains(self,u,i):
        'whether user u rated item i'
        return self.trainingMatrix.contains(self.user[u],self.item[i])


    def row(self,u):
        return self.trainingMatrix.row(self.user[u])

    def col(self,c):
        return self.trainingMatrix.col(self.item[c])

    def rating(self,u,c):
        return self.trainingMatrix.elem(self.user[u],self.item[c])

    def ratingScale(self):
        return (self.rScale[0],self.rScale[1])

    def elemCount(self):
        return self.trainingMatrix.elemCount()









