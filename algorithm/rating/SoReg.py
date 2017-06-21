from baseclass.SocialRecommender import SocialRecommender
from tool import config
from tool import qmath


class SoReg(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(SoReg, self).__init__(conf,trainingSet,testSet,relation,fold)

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
        from collections import defaultdict
        self.Sim = defaultdict(dict)
        print 'constructing similarity matrix...'
        for user in self.dao.user:
            for f in self.sao.getFollowees(user):
                if self.Sim.has_key(user) and self.Sim[user].has_key(f):
                    pass
                else:
                    self.Sim[user][f]=self.sim(user,f)
                    self.Sim[f][user]=self.Sim[user][f]


    def sim(self,u,v):
        return (qmath.pearson_sp(self.dao.sRow(u), self.dao.sRow(v))+self.sao.weight(u,v))/2.0

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                user, item, rating = entry
                uid = self.dao.user[user]
                id = self.dao.item[item]
                simSumf = 0
                simSum2 = 0
                # add the followees' influence
                if self.sao.getFollowees(user):
                    for f in self.sao.getFollowees(user):
                        if self.dao.containsUser(f):
                            fid = self.dao.user[f]
                            simSumf += self.Sim[user][f]*(self.P[uid]-self.P[fid])
                            simSum2 += self.Sim[user][f] * ((self.P[uid] - self.P[fid]).dot(self.P[uid] - self.P[fid]))

                error = rating - self.P[uid].dot(self.Q[id])
                p = self.P[uid]
                q = self.Q[id]

                self.loss += error**2 + self.alpha*simSum2

                #update latent vectors
                self.P[uid] += self.lRate*(error*q - self.regU * p - self.alpha*simSumf)
                self.Q[id] += self.lRate*(error*p - self.regI * q)

            self.loss += self.penaltyLoss()
            iteration += 1
            if self.isConverged(iteration):
                break

    def penaltyLoss(self):
        return self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum()