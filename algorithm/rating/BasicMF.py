from baseclass.IterativeRecommender import IterativeRecommender

class BasicMF(IterativeRecommender):
    def __init__(self,conf):
        super(BasicMF, self).__init__(conf)

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for index,triple in enumerate(self.dao.triple):
                u,i,r = triple
                error = r-self.P[u].dot(self.Q[i])
                self.loss+=error**2
                #update latent vectors
                p = self.P[u].copy()
                q = self.Q[i].copy()
                self.P[u] = p+self.lRate*(error*q-self.regU*p)
                self.Q[i] = q+self.lRate*(error*p-self.regI*q)
            iteration += 1
            self.isConverged(iteration)
