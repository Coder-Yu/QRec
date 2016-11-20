from baseclass.Recommender import Recommender

class ItemMean(Recommender):
    def __init__(self,conf):
        super(ItemMean, self).__init__(conf)

    def predict(self,u,i):
        return self.dao.itemMeans[i]