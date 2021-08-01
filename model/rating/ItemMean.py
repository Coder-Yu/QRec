from base.Recommender import Recommender

class ItemMean(Recommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(ItemMean, self).__init__(conf,trainingSet,testSet,fold)
    def predict(self,u,i):
        if self.data.containsItem(i):
            return self.data.itemMeans[i]
        else:
            return self.data.globalMean