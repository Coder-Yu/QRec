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
                x1 = self.dao.sCol(item)
                x2 = self.dao.sCol(item2)
                diff = 0.0
                commonItem = 0
                for key in x1:
                    if x2.has_key(key):
                        diff+=x1[key]-x2[key]
                        commonItem+=1
                if commonItem==0:
                    diffAverage_sub.setdefault(item2, 0)
                else:
                    diffAverage_sub.setdefault(item2,diff/commonItem)
                freq_sub.setdefault(item2,commonItem)
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

