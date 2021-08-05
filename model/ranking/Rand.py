#coding:utf8
from base.recommender import Recommender
import numpy as np
class Rand(Recommender):
    # Recommend items for every user at random
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(Rand, self).__init__(conf,trainingSet,testSet,fold)

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            return np.random.random(self.data.trainingSize()[1])
        else:
            return [self.data.globalMean] * self.num_items


