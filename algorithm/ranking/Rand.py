#coding:utf8
from baseclass.Recommender import Recommender
import numpy as np
class Rand(Recommender):

    # Recommend items for every user at random

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(Rand, self).__init__(conf,trainingSet,testSet,fold)



    def predict(self,user,item):
        return 0

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            return np.random.random(self.dao.trainingSize()[1])
        else:
            return [self.dao.globalMean] * len(self.dao.item)


