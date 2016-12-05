import sys
from re import split
from tool.config import Config,LineConfig
from tool.file import FileIO
from evaluation.dataSplit import *
from multiprocessing import Process,Manager

class RecQ(object):
    def __init__(self,config):
        self.trainingData = []  # training data
        self.testData = []  # testData
        self.relation = []
        self.measure = []
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

        if config.contains('social'):
            self.socialConfig = LineConfig(self.config['social.setup'])
            self.__loadRelationship(self.config['social'])


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

    def __loadRelationship(self, filePath):
        print 'load social data...'
        with open(filePath) as f:
            relations = f.readlines()
            # ignore the headline
        if self.socialConfig.contains('-header'):
            relations = relations[1:]
        # order of the columns
        order = self.socialConfig['-columns'].strip().split()
        if len(order) <= 2:
            print 'The social file is not in a correct format.'
        for line in relations:
            items = split(' |,|\t', line.strip())
            if len(order) < 2:
                print 'The social file is not in a correct format. Error: Line num %d' % lineNo
                exit(-1)
            userId1 = items[int(order[0])]
            userId2 = items[int(order[1])]
            if len(order) < 3:
                weight = 1
            else:
                weight = float(items[int(order[2])])
            self.relation.append([userId1, userId2, weight])


    def execute(self):
        #import the algorithm module
        importStr = 'from algorithm.rating.' + self.config['recommender'] + ' import ' + self.config['recommender']
        exec (importStr)
        if self.evaluation.contains('-cv'):
            k = int(self.evaluation['-cv'])
            #create the manager used to communication in multiprocess
            manager = Manager()
            m = manager.dict()
            i = 1
            tasks = []
            for train,test in DataSplit.crossValidation(self.trainingData,k):
                fold = '['+str(i)+']'
                if self.config.contains('social'):
                    recommender = self.config['recommender'] + "(self.config,train,test,self.relation,fold)"
                else:
                    recommender = self.config['recommender']+ "(self.config,train,test,fold)"
               #create the process
                p = Process(target=run,args=(m,eval(recommender),i))
                tasks.append(p)
                i+=1
            #start the processes
            for p in tasks:
                p.start()
            #wait until all processes are completed
            for p in tasks:
                p.join()
            #compute the mean error of k-fold cross validation
            self.measure = [dict(m)[i] for i in range(1,k+1)]
            res = []
            for i in range(len(self.measure[0])):
                measure = self.measure[0][i].split(':')[0]
                total = 0
                for j in range(k):
                    total += float(self.measure[j][i].split(':')[1])
                res.append(measure+':'+str(total/k)+'\n')
            #output result
            outDir = LineConfig(self.config['output.setup'])['-dir']
            fileName = self.config['recommender'] +'@'+str(k)+'-fold-cv' + '.txt'
            FileIO.writeFile(outDir,fileName,res)


        else:
            if self.config.contains('social'):
                recommender = self.config['recommender']+'(self.config,self.trainingData,self.testData,self.relation)'
            else:
                recommender = self.config['recommender'] + '(self.config,self.trainingData,self.testData)'
            eval(recommender).execute()


def run(measure,algor,order):
    measure[order] = algor.execute()