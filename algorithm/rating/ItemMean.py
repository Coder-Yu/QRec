from baseclass.Recommender import Recommender

class ItemMean(Recommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(ItemMean, self).__init__(conf,trainingSet,testSet,fold)

    def predict(self,u,i):
        if self.dao.containsItem(i):
            return self.dao.itemMeans[i]
        else:
            return self.dao.globalMean