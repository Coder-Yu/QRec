import numpy as np
from structure import sparseMatrix
from tool.config import Config,LineConfig
import os.path
#from sklearn.cross_validation import train_test_split
class ratingDAO(object):
    'data access control'
    def __init__(self,config):
        self.config = config
        self.ratingConfig = LineConfig(config['ratings.setup'])
        self.evaluation = LineConfig(config['evaluation'])
        self.user = {}
        self.item = {}
        self.timestamp = {}
        self.ratingMatrix = None
        self.trainingMatrix = None
        self.validationMatrix = None
        self.testSet = None
        self.rScale = [-9999999,999999]
        if self.evaluation.contains('-testSet'):
            #specify testSet
            self.trainingMatrix = self.loadRatings(config['ratings'])
            self.testSet = self.loadRatings(self.evaluation['-testSet'],True)

        else: #cross validation and leave-one-out
            self.ratingMatrix = self.loadRatings(config['ratings'])


    def loadRatings(self,file,bTest=False):
        with open(file) as f:
            ratings = f.readlines()
        #ignore the headline
        if self.ratingConfig.contains('-header'):
            ratings = ratings[1:]
        #set delimiter
        delimiter = ' '
        if self.ratingConfig.contains('-d'):
            delimiter = self.ratingConfig['-d']
        #order of the columns
        order = self.ratingConfig['-columns'].strip().split()
        #split data
        userList= []
        u_i_r = {}
        for line in ratings:
            items = line.strip().split(delimiter)
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
            u_i_r[userId].append((float(rating),itemId))

        if not bTest:
            #contruct the sparse matrix
            data=[]
            indices=[]
            indptr=[]
            offset = 0
            for uid in userList:
                uRating = [r[0] for r in u_i_r[uid]]
                uColunms = [self.item[r[1]] for r in u_i_r[uid]]
                data += uRating
                indices += uColunms
                indptr .append(offset)
                offset += len(uRating)
            indptr.append(offset)
            return sparseMatrix.SparseMatrix(data, indices, indptr)
        else:
            # return testSet
            return u_i_r

    def row(self,u):
        return self.trainingMatrix.row(self.user[u])

    def column(self,c):
        return self.trainingMatrix.col(self.item[c])

    def sRow(self,u):
        return self.trainingMatrix.sRow(self.user[u])

    def sColumn(self,c):
        return self.trainingMatrix.sCol(self.item[c])

    def rating(self,u,c):
        return self.trainingMatrix.elem(self.user[u],self.item[c])

    def ratingScale(self):
        return (self.rScale[0],self.rScale[1])









