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
        average = {}
        for i1 in self.dao.item:
            count = 0
            sum = 0
            for n in range(len(self.dao.col(i1))):
                if self.dao.col(i1)[n] != 0:
                    count = count + 1
                    sum = sum + self.dao.col(i1)[n]
                    ave = sum / count
                else:
                    continue
            average.setdefault(i1,ave)klkl

    def predict(self,u,i):
        diff={}
        for i2 in self.dao.item:
            pass
