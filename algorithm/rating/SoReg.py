from baseclass.SocialRecommender import SocialRecommender
from tool import config
from tool import qmath


class SoReg(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(SoReg, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(SoReg, self).readConfiguration()
        alpha = config.LineConfig(self.config['SoReg'])
        self.alpha = float(alpha['-alpha'])

    def printAlgorConfig(self):
        super(SoReg, self).printAlgorConfig()
        print 'Specified Arguments of',self.config['recommender']+':'
        print 'alpha: %.3f' %self.alpha
        print '='*80

    def initModel(self):
        super(SoReg, self).initModel()
        # compute similarity
        self.Sim = {}
        for entry in self.dao.trainingData:
            u, i, r = entry
            if self.sao.getFollowees(u):
                for f in self.sao.getFollowees(u):
                    if not self.Sim.has_key(u):
                        self.Sim[u]={}
                    self.Sim[u][f]=self.sim(u,f)



    def sim(self,u,v):
        return (qmath.pearson(self.dao.row(u), self.dao.row(v))+1)/2.0

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                u, i, r = entry
                uid = self.dao.getUserId(u)
                id = self.dao.getItemId(i)
                simSumf = 0
                simSum2 = 0
                # add the followees' influence
                if self.sao.getFollowees(u):
                    for f in self.sao.getFollowees(u):
                        if self.dao.containsUser(f):
                            fid = self.dao.getUserId(f)
                            simSumf += self.Sim[u][f]*(self.P[uid]-self.P[fid])
                            simSum2 += self.Sim[u][f] * ((self.P[uid] - self.P[fid]).dot(self.P[uid] - self.P[fid]))

                error = r - self.P[uid].dot(self.Q[id])
                p = self.P[uid].copy()
                q = self.Q[id].copy()

                self.loss += error**2 + self.alpha*simSum2 + self.regU * p.dot(p) + self.regI * q.dot(q)

                #update latent vectors
                self.P[uid] += self.lRate*(error*q - self.regU * p - self.alpha*simSumf)
                self.Q[id] += self.lRate*(error*p - self.regI * q)

            iteration += 1
            if self.isConverged(iteration):
                break
