from recommender import Recommender

class IterativeRecommender(Recommender):
    def __init__(self,conf):
        super(IterativeRecommender, self).__init__(conf)

    def initModel(self):
        pass

    def saveModel(self):
        pass

    def loadModel(self):
        pass

    def updateLearningRate(self):
        pass

    def predict(self,u,i):
        pass

    def isConverged(self):
        pass
