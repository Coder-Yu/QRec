from base.recommender import Recommender

class SlopeOne(Recommender,):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(SlopeOne, self).__init__(conf,trainingSet,testSet,fold)
        self.diffAverage = {}
        self.freq = {}

    def initModel(self):
        self.computeAverage()

    def computeAverage(self):
        for item in self.data.testSet_i:
            freq_sub = {}
            diffAverage_sub = {}
            for item2 in self.data.item:
                x1 = self.data.sCol(item)
                x2 = self.data.sCol(item2)
                diff = 0.0
                commonItem = 0
                for key in x1:
                    if key in x2:
                        diff+=x1[key]-x2[key]
                        commonItem+=1
                if commonItem==0:
                    diffAverage_sub.setdefault(item2, 0)
                else:
                    diffAverage_sub.setdefault(item2,diff/commonItem)
                freq_sub.setdefault(item2,commonItem)
            print('item '+ item +" finished.")
            self.diffAverage[item] = diffAverage_sub
            self.freq[item] = freq_sub

    def predictForRating(self, u, i):
        # check if the user existed in trainSet or not
        if self.data.containsUser(u):
            sum = 0
            freqSum = 0
            itemRated,ratings = self.data.userRated(u)
            for item,rating in zip(itemRated,ratings):
                diff = self.diffAverage[i][item]
                count = self.freq[i][item]
                sum += (rating + diff) * count
                freqSum += count
            try:
                pred = float(sum)/freqSum
            except ZeroDivisionError:
                pred = self.data.userMeans[u]
        elif self.data.containsItem(i):
            pred = self.data.itemMeans[i]
        else:
            pred = self.data.globalMean

        return pred

