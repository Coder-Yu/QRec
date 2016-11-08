from baseclass.recommender import Recommender
from tool import qmath
from structure.symmetricMatrix import SymmetricMatrix

class UserKNN(Recommender):
    def __init__(self,conf):
        super(UserKNN, self).__init__(conf)
        super(UserKNN, self).readConfiguration()
        self.userSim = SymmetricMatrix(len(self.dao.user))

    def readConfiguration(self):
        self.sim = self.config['similarity']
        self.shrinkage =int(self.config['num.shrinkage'])
        self.neighbors = int(self.config['num.neighbors'])

    def initModel(self):
        self.computeCorr()


    def predict(self,u,i):
        #find the closest neighbors of user u
        topUsers = sorted(self.userSim[u].iteritems(),key = lambda d:d[1],reverse=True)
        userCount = self.neighbors
        if userCount > len(topUsers):
            userCount = len(topUsers)
        #predict
        pred = 0
        denom = 0
        for n in range(userCount):
            #if user n has rating on item i
            if self.dao.rating(topUsers[n][0],i) != 0:
                corr = topUsers[n][1]
                rating = self.dao.rating(topUsers[n][0],i)
                pred += corr*rating
                denom += topUsers[n][1]
        if pred == 0:
            #no users have rating on item i,return the average rating of user u
            n = self.dao.row(u)>0
            if sum(n[0]) == 0: #no data about current user in training set
                return 0
            pred = float(self.dao.row(u)[0].sum()/n[0].sum())
            return round(pred,3)
        pred = pred/float(denom)
        return round(pred,3)

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
            print u1,'finished.'
        print 'The user correlation has been figured out.'



