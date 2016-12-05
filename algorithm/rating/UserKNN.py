from baseclass.Recommender import Recommender
from tool import qmath
from structure.symmetricMatrix import SymmetricMatrix
import numpy as np

class UserKNN(Recommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(UserKNN, self).__init__(conf,trainingSet,testSet,fold)
        self.userSim = SymmetricMatrix(len(self.dao.user))

    def readConfiguration(self):
        super(UserKNN, self).readConfiguration()
        self.sim = self.config['similarity']
        self.shrinkage =int(self.config['num.shrinkage'])
        self.neighbors = int(self.config['num.neighbors'])

    def printAlgorConfig(self):
        "show algorithm's configuration"
        super(UserKNN, self).printAlgorConfig()
        print 'Specified Arguments of',self.config['recommender']+':'
        print 'num.neighbors:',self.config['num.neighbors']
        print 'num.shrinkage:', self.config['num.shrinkage']
        print 'similarity:', self.config['similarity']
        print '='*80

    def initModel(self):
        self.computeCorr()

    def predict(self,u,i):
        #find the closest neighbors of user u
        topUsers = sorted(self.userSim[u].iteritems(),key = lambda d:d[1],reverse=True)
        userCount = self.neighbors
        if userCount > len(topUsers):
            userCount = len(topUsers)
        #predict
        sum,denom = 0,0
        for n in range(userCount):
            #if user n has rating on item i
            similarUser = topUsers[n][0]
            if self.dao.rating(similarUser,i) != 0:
                similarity = topUsers[n][1]
                rating = self.dao.rating(similarUser,i)
                sum += similarity*(rating-self.dao.userMeans[similarUser])
                denom += similarity
        if sum == 0:
            #no users have rating on item i,return the average rating of user u
            if not self.dao.containsUser(u):
                #user u has no ratings in the training set,return the global mean
                return self.dao.globalMean
            return self.dao.userMeans[u]
        pred = self.dao.userMeans[u]+sum/float(denom)
        return pred

    def computeCorr(self):
        'compute correlation among users'
        print 'Computing user correlation...'
        for u1 in self.dao.testSet_u:

            for u2 in self.dao.user:
                if u1 <> u2:
                    if self.userSim.contains(u1,u2):
                        continue
                    sim = qmath.similarity(self.dao.row(u1),self.dao.row(u2),self.sim)
                    self.userSim.set(u1,u2,sim)
            print 'user '+u1+' finished.'
        print 'The user correlation has been figured out.'



