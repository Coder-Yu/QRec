#coding:utf8
from base.recommender import Recommender
import numpy as np
class MostPopular(Recommender):
    # Recommend the most popular items for every user
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(MostPopular, self).__init__(conf,trainingSet,testSet,fold)

    def initModel(self):
        self.popularItemList = np.random.random(self.data.trainingSize()[1])
        for item in self.data.trainSet_i:
            ind = self.data.item[item]
            self.popularItemList[ind] = len(self.data.trainSet_i[item])

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            return self.popularItemList
        else:
            return [self.data.globalMean] * self.num_items


