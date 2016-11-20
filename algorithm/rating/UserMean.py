from baseclass.Recommender import Recommender

class UserMean(Recommender):
    def __init__(self,conf):
        super(UserMean, self).__init__(conf)

    def predict(self,u,i):
        return self.dao.userMeans[u]