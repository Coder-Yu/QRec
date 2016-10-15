from baseclass.recommender import Recommender

class UserKNN(Recommender):
    def __init__(self,rMatrix,configuration):
        super(UserKNN, self).__init__(rMatrix,configuration)


    def computeCorr(self):
        pass

    def predict(self,u,i):
        pass