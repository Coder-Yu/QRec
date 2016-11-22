from baseclass.IterativeRecommender import IterativeRecommender

class SVD(IterativeRecommender):
    def __init__(self,conf):
        super(SVD, self).__init__(conf)

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            self.bi = {}
            self.bu = {}
            for triple in self.dao.triple:
                u,i,r = triple
                self.bi[i] = self.dao.itemMeans[i]-self.dao.globalMean
                self.bu[u] = self.dao.userMeans[u]-self.dao.globalMean
                error = r-self.P[u].dot(self.Q[i])-self.dao.globalMean-self.bi[i]-self.bu[u]
                self.loss+=error**2
                #update latent vectors
                p = self.P[u].copy()
                q = self.Q[i].copy()

                self.bi[i] = self.bi[i]+self.lRate*(error-self.regI*self.bi[i])

                self.P[u] = p+self.lRate*(error*q-self.regU*p)
                self.Q[i] = q+self.lRate*(error*p-self.regI*q)


            iteration += 1
            self.isConverged(iteration)

    def predict(self,u,i):
        if self.dao.containsUser(u) and self.dao.containsItem(i):
            return self.P[self.dao.user[u]].dot(self.Q[self.dao.item[i]])+self.dao.globalMean+self.bi[self.dao.item[i]]+self.bu[self.dao.user[u]]
        else:
            return self.dao.globalMean
