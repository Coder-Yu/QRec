from baseclass.Recommender import Recommender
from tool import qmath
from structure.symmetricMatrix import SymmetricMatrix

class SlopeOne(Recommender,):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(SlopeOne, self).__init__(conf,trainingSet,testSet,fold)
        self.diffAverage = {}
        self.freq = {}

    def initModel(self):
        self.computeAverage()

    def computeAverage(self):
        for item in self.dao.testSet_i:
            freq_sub = {}
            diffAverage_sub = {}
            for item2 in self.dao.item:
                new_x1,new_x2 = qmath.common(self.dao.col(item),self.dao.col(item2))
                diff = new_x1 - new_x2
                if len(diff)==0:
                    diffAverage_sub.setdefault(self.dao.item[item2], 0)
                else:
                    diffAverage_sub.setdefault(self.dao.item[item2],diff.mean())
                freq_sub.setdefault(self.dao.item[item2],len(diff))
            print 'item '+ item +" finished."
            self.diffAverage[item] = diffAverage_sub
            self.freq[item] = freq_sub


    def predict(self,u,i):
        pred = 0
        # check if the user existed in trainSet or not
        if self.dao.containsUser(u):
            sum = 0
            freqSum = 0
            itemRated,ratings = self.dao.userRated(u)
            for item,rating in zip(itemRated,ratings):
                diff = self.diffAverage[i][item]
                count = self.freq[i][item]
                sum += (rating + diff) * count
                freqSum += count
            try:
                pred = float(sum)/freqSum
            except ZeroDivisionError:
                pred = self.dao.userMeans[u]
        elif self.dao.containsItem(i):
            pred = self.dao.itemMeans[i]
        else:
            pred = self.dao.globalMean

        return pred

