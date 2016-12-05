from random import random
from tool.file import FileIO
class DataSplit(object):

    def __init__(self):
        pass

    @staticmethod
    def dataSplit(data,test_ratio = 0.3,output=False,path='./',order=1):
        testSet = []
        trainingSet = []
        for entry in data:
            if random() < test_ratio:
                testSet.append(entry)
            else:
                trainingSet.append(entry)

        if output:
            FileIO.writeFile(path,'testSet['+str(order)+']',testSet)
            FileIO.writeFile(path, 'trainingSet[' + str(order) + ']', trainingSet)
        return trainingSet,testSet

    @staticmethod
    def crossValidation(data,k,output=False,path='./',order=1):
        if k<=1 or k>10:
            k=3
        for i in range(k):
            trainingSet = []
            testSet = []
            for ind,line in enumerate(data):
                if ind%k == i:
                    testSet.append(line[:])
                else:
                    trainingSet.append(line[:])
            yield trainingSet,testSet


