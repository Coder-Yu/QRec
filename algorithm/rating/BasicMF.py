from baseclass.IterativeRecommender import IterativeRecommender

class BasicMF(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(BasicMF, self).__init__(conf,trainingSet,testSet,fold)

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                u, i, r = entry
                u = self.dao.getUserId(u)
                i = self.dao.getItemId(i)
                error = r - self.P[u].dot(self.Q[i])
                self.loss += error**2
                p = self.P[u].copy()
                q = self.Q[i].copy()

                #update latent vectors
                self.P[u] += self.lRate*error*q
                self.Q[i] += self.lRate*error*p


            iteration += 1
            if self.isConverged(iteration):
                break
