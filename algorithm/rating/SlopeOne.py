from baseclass.recommender import Recommender
from tool import qmath
from structure.symmetricMatrix import SymmetricMatrix

class SlopeOne(Recommender):
    def __init__(self,conf):
        super(SlopeOne, self).__init__(conf)
        pass

    def initModel(self):
        self.computeAverage()

    def computeAverage(self):
        diffAverage = {}
        freq = {}
        for i1 in range(len(self.dao.testSet_i.keys())):
            for i2 in self.dao.testSet_i.keys()[i1:]:
                new_x1,new_x2 = qmath.common(self.dao.testSet_i.keys()[i1],i2)
                diff = new_x1 - new_x2
                diffAverage_sub = {}
                diffAverage_sub.setdefault(i2,diff.sum()/len(diff))

                freq_sub = {}
                freq_sub.setdefault(i2,len(diff))

            diffAverage.setdefault(self.dao.testSet_i.keys()[i1],diffAverage_sub)
            freq.setdefault(self.dao.testSet_i.keys()[i1],freq_sub)


    def predict(self,u,i):
        for u in self.dao.testSet_u.keys():
            if SymmetricMatrix.contains(u) == True:#should use another interface
                itemList = self.dao.row(u)>0
                for item in self.dao.testSet_u[u].keys():
                    pass
            else:
                for item in self.dao.testSet_u[u].keys():
                    pred = self.dao.itemMeans[item]

