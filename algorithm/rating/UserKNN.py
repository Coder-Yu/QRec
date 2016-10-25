from baseclass.recommender import Recommender

class UserKNN(Recommender):
    def __init__(self,conf):
        super(UserKNN, self).__init__(conf)



    def computeCorr(self):
        pass

    def predict(self,u,i):
        pass