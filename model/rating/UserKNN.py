from base.recommender import Recommender
from util import qmath
from util.structure.symmetricMatrix import SymmetricMatrix


class UserKNN(Recommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(UserKNN, self).__init__(conf,trainingSet,testSet,fold)
        self.userSim = SymmetricMatrix(len(self.data.user))

    def readConfiguration(self):
        super(UserKNN, self).readConfiguration()
        self.sim = self.config['similarity']
        self.neighbors = int(self.config['num.neighbors'])

    def printAlgorConfig(self):
        "show model's configuration"
        super(UserKNN, self).printAlgorConfig()
        print('Specified Arguments of',self.config['model.name']+':')
        print('num.neighbors:',self.config['num.neighbors'])
        print('similarity:', self.config['similarity'])
        print('='*80)

    def initModel(self):
        self.topUsers = {}
        self.computeSimilarities()

    def predictForRating(self, u, i):
        #find the closest neighbors of user u
        topUsers = self.topUsers[u]
        userCount = self.neighbors
        if userCount > len(topUsers):
            userCount = len(topUsers)
        #predict
        sum,denom = 0,0
        for n in range(userCount):
            #if user n has rating on item i
            similarUser = topUsers[n][0]
            if self.data.rating(similarUser,i) != -1:
                similarity = topUsers[n][1]
                rating = self.data.rating(similarUser,i)
                sum += similarity*(rating-self.data.userMeans[similarUser])
                denom += similarity
        if sum == 0:
            #no users have rating on item i,return the average rating of user u
            if not self.data.containsUser(u):
                #user u has no ratings in the training set,return the global mean
                return self.data.globalMean
            return self.data.userMeans[u]
        pred = self.data.userMeans[u]+sum/float(denom)
        return pred

    def computeSimilarities(self):
        'compute correlation among users'
        print('Computing user similarities...')
        for idx,u1 in enumerate(self.data.testSet_u):
            for u2 in self.data.user:
                if u1 != u2:
                    if self.userSim.contains(u1,u2):
                        continue
                    sim = qmath.similarity(self.data.sRow(u1),self.data.sRow(u2),self.sim)
                    self.userSim.set(u1,u2,sim)
            self.topUsers[u1]=sorted(iter(self.userSim[u1].items()), key=lambda d: d[1], reverse=True)
            if idx%100==0:
                print('progress:',idx,'/',len(self.data.testSet_u))
        print('The user similarities have been calculated.')

    def predictForRanking(self,u):
        print('Using Memory based algorithms to rank items is extremely time-consuming. So ranking for all items in UserKNN is not available.')
        exit(0)
