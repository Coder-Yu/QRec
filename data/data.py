import numpy as np
from structure import sparseMatrix
from tool.config import Config,LineConfig
import os.path
class DAO(object):
    'data access control'
    def __init__(self,config):
        self.config = config
        self.ratingConfig = LineConfig(config['ratings.setup'])
        self.user = {}
        self.item = {}
        self.ratings = {}
        self.timestamp = {}
        self.trainingMatrix = None
        self.validationMatrix = None
        self.testMatrix = None


    def loadTrainingSet(self):
        if not os.path.exists(self.config['dataset.ratings']):
            print 'ratings file is not found!'
            exit()
        with open(self.config['dataset.ratings']) as f:
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
        for line in ratings:
            items = line.strip().split(delimiter)
            userId =  items[int(order[0])]
            itemId =  items[int(order[1])]
            rating =  items[int(order[2])]
            #order the user
            if not self.user.has_key(userId):
                self.ratings[userId] = []
                self.user[userId] = len(self.user)
                userList.append(userId)
            #order the item
            if not self.item.has_key(itemId):
                self.item[itemId] = len(self.item)
            self.ratings[userId].append((float(rating),self.item[itemId]))
        #contruct the sparse matrix
        data=[]
        indices=[]
        indptr=[]
        offset = 0
        for uid in userList:
            uRating = [r[0] for r in self.ratings[uid]]
            uColunms = [r[1] for r in self.ratings[uid]]
            data += uRating
            indices += uColunms
            indptr .append(offset)
            offset += len(uRating)
        indptr.append(offset)
        self.trainingMatrix = sparseMatrix.SparseMatrix(data,indices,indptr)


    def loadTestSet(self):
        if not os.path.exists(self.config['dataset.ratings']):
            print 'ratings file is not found!'
            exit()
        with open(self.config['dataset.ratings']) as f:
            ratings = f.readlines()
        # ignore the headline
        if self.ratingConfig.contains('-header'):
            ratings = ratings[1:]
        # set delimiter
        delimiter = ' '
        if self.ratingConfig.contains('-d'):
            delimiter = self.ratingConfig['-d']
        # order of the columns
        order = self.ratingConfig['-columns'].strip().split()
        # split data
        userList = []
        for line in ratings:
            items = line.strip().split(delimiter)
            userId = items[int(order[0])]
            itemId = items[int(order[1])]
            rating = items[int(order[2])]
            # order the user
            if not self.user.has_key(userId):
                self.ratings[userId] = []
                self.user[userId] = len(self.user)
                userList.append(userId)
            # order the item
            if not self.item.has_key(itemId):
                self.item[itemId] = len(self.item)
            self.ratings[userId].append((float(rating), self.item[itemId]))
        # contruct the sparse matrix
        data = []
        indices = []
        indptr = []
        offset = 0
        for uid in userList:
            uRating = [r[0] for r in self.ratings[uid]]
            uColunms = [r[1] for r in self.ratings[uid]]
            data += uRating
            indices += uColunms
            indptr.append(offset)
            offset += len(uRating)
        indptr.append(offset)
        self.trainingMatrix = sparseMatrix.SparseMatrix(data, indices, indptr)


    def row(self,u):
        return self.trainingMatrix.row(u)


c = Config('../config/UserKNN')
d = DAO(c)
d.loadTrainingSet()
print d.trainingMatrix.toDense()





