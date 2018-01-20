#coding:utf8
from baseclass.Recommender import Recommender
import numpy as np
class MostPopular(Recommender):

    # Recommend the most popular items for every user

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(MostPopular, self).__init__(conf,trainingSet,testSet,fold)

    # def readConfiguration(self):
    #     super(BPR, self).readConfiguration()

    def initModel(self):
        self.popularItemList = np.random.random(self.dao.trainingSize()[1])
        for item in self.dao.trainSet_i:
            ind = self.dao.item[item]
            self.popularItemList[ind] = len(self.dao.trainSet_i[item])


    def predict(self,user,item):
        return 0

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            return self.popularItemList
        else:
            return [self.dao.globalMean] * len(self.dao.item)


