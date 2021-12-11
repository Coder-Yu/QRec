from base.socialRecommender import SocialRecommender
from util import config
from util import qmath


class SoReg(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(SoReg, self).__init__(conf,trainingSet,testSet,relation,fold)

    def readConfiguration(self):
        super(SoReg, self).readConfiguration()
        alpha = config.OptionConf(self.config['SoReg'])
        self.alpha = float(alpha['-alpha'])

    def printAlgorConfig(self):
        super(SoReg, self).printAlgorConfig()
        print('Specified Arguments of',self.config['model.name']+':')
        print('alpha: %.3f' %self.alpha)
        print('='*80)

    def initModel(self):
        super(SoReg, self).initModel()
        # compute similarity
        from collections import defaultdict
        self.Sim = defaultdict(dict)
        print('constructing similarity matrix...')
        for user in self.data.user:
            for f in self.social.getFollowees(user):
                if user in self.Sim and f in self.Sim[user]:
                    pass
                else:
                    self.Sim[user][f]=self.sim(user,f)
                    self.Sim[f][user]=self.Sim[user][f]

    def sim(self,u,v):
        return (qmath.pearson_sp(self.data.sRow(u), self.data.sRow(v))+self.social.weight(u,v))/2.0

    def trainModel(self):
        epoch = 0
        while epoch < self.maxEpoch:
            self.loss = 0
            for entry in self.data.trainingData:
                user, item, rating = entry
                uid = self.data.user[user]
                vid = self.data.item[item]
                # add the followees' influence
                error = rating - self.P[uid].dot(self.Q[vid])
                p = self.P[uid]
                q = self.Q[vid]
                self.loss += error**2
                #update latent vectors
                self.P[uid] += self.lRate*(error*q - self.regU * p)
                self.Q[vid] += self.lRate*(error*p - self.regI * q)
            for user in self.social.user:
                simSum = 0
                simSumf1 = 0
                if not self.data.containsUser(user):
                    continue
                uid = self.data.user[user]
                for f in self.social.getFollowees(user):
                    if self.data.containsUser(f):
                        fid = self.data.user[f]
                        simSumf1 += self.Sim[user][f] * (self.P[uid] - self.P[fid])
                        simSum += self.Sim[user][f] * ((self.P[uid] - self.P[fid]).dot(self.P[uid] - self.P[fid]))
                        self.loss += simSum
                simSumf2 = 0
                for g in self.social.getFollowers(user):
                    if self.data.containsUser(g):
                        gid = self.data.user[g]
                        simSumf2 += self.Sim[user][g] * (self.P[uid]-self.P[gid])

                self.P[uid] += self.lRate * (- self.alpha * (simSumf1+simSumf2))

            self.loss += self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum()
            epoch += 1
            if self.isConverged(epoch):
                break
