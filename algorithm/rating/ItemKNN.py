from baseclass.Recommender import Recommender
from tool import qmath
from structure.symmetricMatrix import SymmetricMatrix

class ItemKNN(Recommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(ItemKNN, self).__init__(conf,trainingSet,testSet,fold)
        self.itemSim = SymmetricMatrix(len(self.dao.user)) #used to store the similarity among items

    def readConfiguration(self):
        super(ItemKNN, self).readConfiguration()
        self.sim = self.config['similarity']
        self.shrinkage =int(self.config['num.shrinkage'])
        self.neighbors = int(self.config['num.neighbors'])

    def printAlgorConfig(self):
        "show algorithm's configuration"
        super(ItemKNN, self).printAlgorConfig()
        print 'Specified Arguments of',self.config['recommender']+':'
        print 'num.neighbors:',self.config['num.neighbors']
        print 'num.shrinkage:', self.config['num.shrinkage']
        print 'similarity:', self.config['similarity']
        print '='*80

    def initModel(self):
        self.computeCorr()

    def predict(self,u,i):
        #find the closest neighbors of item i
        topItems = sorted(self.itemSim[i].iteritems(),key = lambda d:d[1],reverse=True)
        itemCount = self.neighbors
        if itemCount > len(topItems):
            itemCount = len(topItems)
        #predict
        sum = 0
        denom = 0
        for n in range(itemCount):
            similarItem = topItems[n][0]
            #if user n has rating on item i
            if self.dao.contains(u,similarItem):
                similarity = topItems[n][1]
                rating = self.dao.rating(u,similarItem)
                sum += similarity*(rating-self.dao.itemMeans[similarItem])
                denom += similarity
        if sum == 0:
            #no items have rating on item i,return the average rating of item i
            if not self.dao.containsItem(i):
                # item i has no ratings in the training set
                return self.dao.globalMean
            return self.dao.itemMeans[i]
        pred = self.dao.itemMeans[i]+sum/float(denom)
        return pred

    def computeCorr(self):
        'compute correlation among items'
        print 'Computing item correlation...'
        for i1 in self.dao.testSet_i:

            for i2 in self.dao.item:
                if i1 <> i2:
                    if self.itemSim.contains(i1,i2):
                        continue
                    sim = qmath.similarity(self.dao.col(i1),self.dao.col(i2),self.sim)
                    self.itemSim.set(i1,i2,sim)
            print 'item '+i1+' finished.'
        print 'The item correlation has been figured out.'



