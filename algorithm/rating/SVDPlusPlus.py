from baseclass.IterativeRecommender import IterativeRecommender

class SVDPlusPlus(IterativeRecommender):
    def __init__(self,conf):
        super(SVDPlusPlus, self).__init__(conf)

    def readConfiguration(self):
        super(SVDPlusPlus, self).readConfiguration()

    def printAlgorConfig(self):
        super(SVDPlusPlus, self).printAlgorConfig()

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for u in self.dao.user:
                for i in self.dao.item:
                    if self.dao.contains(u,i):
                        rating = self.dao.rating(u,i)
                        error = rating-self.predict(u,i)
                        self.loss+=error**2
            self.isConverged(iteration)
