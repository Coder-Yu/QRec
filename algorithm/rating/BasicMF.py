from baseclass.IterativeRecommender import IterativeRecommender

class BasicMF(IterativeRecommender):
    def __init__(self,conf):
        super(BasicMF, self).__init__(conf)

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for triple in self.dao.triple:
                u,i,r = triple
                u = self.dao.getUserId(u)
                i = self.dao.getItemId(i)
                error = r - self.P[u].dot(self.Q[i])
                self.loss += error**2
                p = self.P[u].copy()
                q = self.Q[i].copy()
                self.loss += self.regU * p.dot(p) + self.regI * q.dot(q)

                #update latent vectors
                self.P[u] += self.lRate*(error*q-self.regU*p)
                self.Q[i] += self.lRate*(error*p-self.regI*q)


            iteration += 1
            if self.isConverged(iteration):
                break
