from base.recommender import Recommender
from util import qmath
from util.structure.symmetricMatrix import SymmetricMatrix

class ItemKNN(Recommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(ItemKNN, self).__init__(conf,trainingSet,testSet,fold)
        self.itemSim = SymmetricMatrix(len(self.data.user)) #used to store the similarity among items

    def readConfiguration(self):
        super(ItemKNN, self).readConfiguration()
        self.sim = self.config['similarity']
        self.neighbors = int(self.config['num.neighbors'])

    def printAlgorConfig(self):
        "show model's configuration"
        super(ItemKNN, self).printAlgorConfig()
        print('Specified Arguments of',self.config['model.name']+':')
        print('num.neighbors:',self.config['num.neighbors'])
        print('similarity:', self.config['similarity'])
        print('='*80)

    def initModel(self):
        self.topItems = {}
        self.computeSimilarities()

    def predictForRating(self, u, i):
        #find the closest neighbors of item i
        topItems = self.topItems[i]
        itemCount = self.neighbors
        if itemCount > len(topItems):
            itemCount = len(topItems)
        #predict
        sum = 0
        denom = 0
        for n in range(itemCount):
            similarItem = topItems[n][0]
            #if user n has rating on item i
            if self.data.contains(u,similarItem):
                similarity = topItems[n][1]
                rating = self.data.rating(u,similarItem)
                sum += similarity*(rating-self.data.itemMeans[similarItem])
                denom += similarity
        if sum == 0:
            #no items have rating on item i,return the average rating of item i
            if not self.data.containsItem(i):
                # item i has no ratings in the training set
                return self.data.globalMean
            return self.data.itemMeans[i]
        pred = self.data.itemMeans[i]+sum/float(denom)
        return pred

    def computeSimilarities(self):
        'compute correlation among items'
        print('Computing item similarities...')
        for idx,i1 in enumerate(self.data.testSet_i):

            for i2 in self.data.item:
                if i1 != i2:
                    if self.itemSim.contains(i1,i2):
                        continue
                    sim = qmath.similarity(self.data.sCol(i1),self.data.sCol(i2),self.sim)
                    self.itemSim.set(i1,i2,sim)
            self.topItems[i1] = sorted(iter(self.itemSim[i1].items()),key = lambda d:d[1],reverse=True)
            if idx%100==0:
                print('progress:',idx,'/',len(self.data.testSet_i))
        print('The item similarities have been calculated.')

    def predictForRanking(self,u):
        print('Using Memory based algorithms to rank items is extremely time-consuming. So ranking for all items in ItemKNN is not available.')
        exit(0)
