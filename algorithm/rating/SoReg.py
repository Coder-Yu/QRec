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
                vid = self.dao.item[item]

                # add the followees' influence

                error = rating - self.P[uid].dot(self.Q[vid])
                p = self.P[uid]
                q = self.Q[vid]

                self.loss += error**2

                #update latent vectors
                self.P[uid] += self.lRate*(error*q - self.regU * p)
                self.Q[vid] += self.lRate*(error*p - self.regI * q)

                simSum = 0
                simSumf1 = 0
                for f in self.sao.getFollowees(user):
                    if self.dao.containsUser(f):
                        fid = self.dao.user[f]
                        simSumf1 += self.Sim[user][f] * (self.P[uid] - self.P[fid])
                        simSum += self.Sim[user][f] * ((self.P[uid] - self.P[fid]).dot(self.P[uid] - self.P[fid]))
                        self.loss += simSum

                simSumf2 = 0
                for g in self.sao.getFollowers(user):
                    if self.dao.containsUser(g):
                        gid = self.dao.user[g]
                        simSumf2 += self.Sim[user][g] * (self.P[uid]-self.P[gid])

                self.P[uid] += self.lRate * (- self.alpha * (simSumf1+simSumf2))

            self.loss += self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum()
            iteration += 1
            if self.isConverged(iteration):
                break
