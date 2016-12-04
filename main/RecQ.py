import sys
sys.path.append("..")
from re import split
from tool.config import Config,LineConfig
from evaluation.dataSplit import *
import algorithm.rating as ar
from algorithm.rating import *
# import os
# from importlib import import_module
# for rt,dir,module in os.walk('../algorithm/rating'):
#     for file in module:
#         index = file.index('.')
#         name = file[:index]
#         print name
#         import_module('/algorithm/rating/'+name)
class RecQ(object):
    def __init__(self,config):
        self.trainingData = []  # training data
        self.testData = []  # testData
        self.config =config
        self.ratingConfig = LineConfig(config['ratings.setup'])
        if self.config.contains('evaluation.setup'):
            self.evaluation = LineConfig(config['evaluation.setup'])
            if self.evaluation.contains('-testSet'):
                #specify testSet
                self.__loadDataSet(config['ratings'])
                self.__loadDataSet(self.evaluation['-testSet'],bTest=True)
            elif self.evaluation.contains('-ap'):
                #auto partition
                self.__loadDataSet(config['ratings'])
                self.trainingData,self.testData = DataSplit.\
                    dataSplit(self.trainingData,test_ratio=float(self.evaluation['-ap']))
            elif self.evaluation.contains('-cv'):
                #cross validation
                self.__loadDataSet(config['ratings'])
                #self.trainingData,self.testData = DataSplit.crossValidation(self.trainingData,int(self.evaluation['-cv']))
            else:
                print 'Evaluation is not well configured!'
                exit(-1)

    def __loadDataSet(self, file, bTest=False):
        if not bTest:
            print 'loading training data...'
        else:
            print 'loading test data...'
        with open(file) as f:
            ratings = f.readlines()
        # ignore the headline
        if self.ratingConfig.contains('-header'):
            ratings = ratings[1:]
        # order of the columns
        order = self.ratingConfig['-columns'].strip().split()

        for lineNo, line in enumerate(ratings):
            items = split(' |,|\t', line.strip())
            if len(order) < 3:
                print 'The rating file is not in a correct format. Error: Line num %d' % lineNo
                exit(-1)
            try:
                userId = items[int(order[0])]
                itemId = items[int(order[1])]
                rating = items[int(order[2])]
            except ValueError:
                print 'Error! Have you added the option -header to the rating.setup?'
                exit(-1)
            if not bTest:
                self.trainingData.append([userId, itemId, float(rating)])
            else:
                self.testData.append([userId, itemId, float(rating)])

    def execute(self):
        self.algor = []
        if self.evaluation.contains('-cv'):
            i = 1
            for train,test in DataSplit.crossValidation(self.trainingData,int(self.evaluation['-cv'])):
                fold = '['+str(i)+']'
                recommender = 'ar.' + self.config['recommender'] + '.' + self.config['recommender']\
                              + "(self.config,train,test,fold)"
                eval(recommender).execute()
                i+=1
        else:
            recommender = 'ar.'+self.config['recommender']+'.'+self.config['recommender']\
                          +'(self.config,self.trainingData,self.testData)'
            eval(recommender).execute()


