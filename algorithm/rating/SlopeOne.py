from baseclass.recommender import Recommender
from tool import qmath
from structure.symmetricMatrix import SymmetricMatrix

class SlopeOne(Recommender):
    def __init__(self,conf):
        super(SlopeOne, self).__init__(conf)
        self.diffAverage = {}
        self.freq = {}

    def initModel(self):
        self.computeAverage()

    def computeAverage(self):
        for i1 in range(len(self.dao.testSet_i.keys())):
            freq_sub = {}
            diffAverage_sub = {}
            for i2 in self.dao.testSet_i.keys()[i1:]:
                new_x1,new_x2 = qmath.common(self.dao.testSet_i.keys()[i1],i2)
                diff = new_x1 - new_x2

                diffAverage_sub.setdefault(i2,diff.sum()/len(diff))
                freq_sub.setdefault(i2,len(diff))

            self.diffAverage[i1] = diffAverage_sub
            self.freq[i1] = freq_sub


    def predict(self,u,i):
        itemDict = {}
        # check if the user existed in trainSet or not
        if self.dao.containsUser(u) == True:
            for item in self.dao.row(u).valuse():
                if item.values() > 0:
                    itemDict[item] = item.values()
                else:
                    continue
            Sum = 0
            freqSum = 0
            for item2 in itemDict.keys():
                Sum = Sum + ((itemDict[item2] + self.diffAverage[u][item2]) * self.freq[u][item2])
                freqSum = freqSum + self.freq[u][item2]
            pred = Sum/freqSum

        else:
            pred = self.dao.itemMeans[u]

