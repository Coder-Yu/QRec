from baseclass.IterativeRecommender import IterativeRecommender

class PMF(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(PMF, self).__init__(conf,trainingSet,testSet,fold)

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                user, item, rating = entry
                u = self.dao.user[user] #get user id
                i = self.dao.item[item] #get item id
                error = rating - self.P[u].dot(self.Q[i])
                self.loss += error**2
                p = self.P[u]
                q = self.Q[i]

                #update latent vectors
                self.P[u] += self.lRate*(error*q-self.regU*p)
                self.Q[i] += self.lRate*(error*p-self.regI*q)

            self.loss += self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum()
            iteration += 1
            if self.isConverged(iteration):
                break
        import matplotlib.pyplot as plt
        x = []
        y = []
        for user in self.dao.user:
            x.append(self.P[self.dao.user[user]][2])
            y.append(self.P[self.dao.user[user]][4])
        print len(x)
        plt.scatter(x, y, marker='*', color='red')
        x = []
        y = []
        for item in self.dao.item:
            x.append(self.Q[self.dao.item[item]][2])
            y.append(self.Q[self.dao.item[item]][4])
        print len(x)
        plt.scatter(x, y, marker='o', color='green')
        plt.show()
