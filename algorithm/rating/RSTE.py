from baseclass.SocialRecommender import SocialRecommender
from tool import config
class RSTE(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(RSTE, self).__init__(conf,trainingSet,testSet,relation,fold)

    def readConfiguration(self):
        super(RSTE, self).readConfiguration()
        alpha = config.LineConfig(self.config['RSTE'])
        self.alpha = float(alpha['-alpha'])

    def printAlgorConfig(self):
        super(RSTE, self).printAlgorConfig()
        print 'Specified Arguments of',self.config['recommender']+':'
        print 'alpha: %.3f' %self.alpha
        print '='*80

    def initModel(self):
        super(RSTE, self).initModel()

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                u, i, r = entry
                error = r - self.predict(u,i)
                i = self.dao.getItemId(i)
                u = self.dao.getUserId(u)
                self.loss += error ** 2
                p = self.P[u].copy()
                q = self.Q[i].copy()
                self.loss += self.regU * p.dot(p) + self.regI * q.dot(q)
                # update latent vectors
                self.P[u] += self.lRate * (self.alpha*error * q - self.regU * p)
                self.Q[i] += self.lRate * (self.alpha*error * p - self.regI * q)
            iteration += 1
            self.isConverged(iteration)

    def predict(self,u,i):
        if self.dao.containsUser(u) and self.dao.containsItem(i):   
            i = self.dao.getItemId(i)
            fPred = 0
            denom = 0
            relations = self.sao.getFollowees(u)
            for followee in relations:
                weight = relations[followee]
                uf = self.dao.getUserId(followee)
                if uf <> -1 and self.dao.containsUser(followee):  # followee is in rating set
                    fPred += weight * (self.P[uf].dot(self.Q[i]))
                    denom += weight
            u = self.dao.getUserId(u)
            if denom <> 0:
                return self.alpha * self.P[u].dot(self.Q[i])+(1-self.alpha)*fPred / denom
            else:
                return self.P[u].dot(self.Q[i])
        else:
            return self.dao.globalMean