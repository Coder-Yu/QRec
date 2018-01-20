from baseclass.Recommender import Recommender

class UserMean(Recommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(UserMean, self).__init__(conf,trainingSet,testSet,fold)

    def predict(self,u,i):
        if self.dao.containsUser(u):
            return self.dao.userMeans[u]
        else:
            return self.dao.globalMean