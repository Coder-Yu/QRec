from baseclass.recommender import Recommender
from tool import qmath
from structure.symmetricMatrix import SymmetricMatrix

class ItemKNN(Recommender):
    def __init__(self,conf):
        super(ItemKNN, self).__init__(conf)
        super(ItemKNN, self).readConfiguration()
        self.itemSim = SymmetricMatrix(len(self.dao.user))

    def readConfiguration(self):
        self.sim = self.config['similarity']
        self.shrinkage =int(self.config['num.shrinkage'])
        self.neighbors = int(self.config['num.neighbors'])

    def initModel(self):
        self.computeCorr()


    def predict(self,u,i):
        #find the closest neighbors of user u
        topItems = sorted(self.itemSim[i].iteritems(),key = lambda d:d[1],reverse=True)
        itemCount = self.neighbors
        if itemCount > len(topItems):
            itemCount = len(topItems)
        #predict
        pred = 0
        denom = 0
        for n in range(itemCount):
            #if user n has rating on item i
            if self.dao.rating(u,topItems[n][0]) != 0:
                corr = topItems[n][1]
                rating = self.dao.rating(u,topItems[n][0])
                pred += corr*rating
                denom += topItems[n][1]
        if pred == 0:
            #no users have rating on item i,return the average rating of user u
            n = self.dao.col(i)>0
            if n[0].sum()== 0: #no data about current user in training set
                return 0
            pred = float(self.dao.col(i)[0].sum()/n[0].sum())
            return round(pred,3)
        pred = pred/float(denom)
        return round(pred,3)

    def computeCorr(self):
        'compute correlation among users'
        print 'Computing item correlation...'
        for i1 in self.dao.testSet_i:

            for i2 in self.dao.item:
                if i1 <> i2:
                    if self.itemSim.contains(i1,i2):
                        continue
                    sim = qmath.similarity(self.dao.col(i1),self.dao.col(i2),self.sim)
                    self.itemSim.set(i1,i2,sim)
            print i1,'finished.'
        print 'The item correlation has been figured out.'



