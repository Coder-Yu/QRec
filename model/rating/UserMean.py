from base.recommender import Recommender

class UserMean(Recommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(UserMean, self).__init__(conf,trainingSet,testSet,fold)

    def predictForRating(self, u, i):
        if self.data.containsUser(u):
            return self.data.userMeans[u]
        else:
            return self.data.globalMean