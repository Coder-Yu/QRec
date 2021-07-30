# QRec is a free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
from data.rating import RatingDAO
from tool.file import FileIO
from tool.config import LineConfig
from tool.log import Log
from os.path import abspath
from time import strftime,localtime,time
from evaluation.measure import Measure
from tool.qmath import find_k_largest


class Recommender(object):
    def __init__(self,conf,trainingSet,testSet,fold='[1]'):
        self.config = conf
        self.data = None
        self.isSaveModel = False
        self.ranking = None
        self.isLoadModel = False
        self.output = None
        self.isOutput = True
        self.data = RatingDAO(self.config, trainingSet, testSet)
        self.foldInfo = fold
        self.evalSettings = LineConfig(self.config['evaluation.setup'])
        self.measure = []
        self.recOutput = []
        if self.evalSettings.contains('-cold'):
            #evaluation on cold-start users
            threshold = int(self.evalSettings['-cold'])
            removedUser = {}
            for user in self.data.testSet_u:
                if user in self.data.trainSet_u and len(self.data.trainSet_u[user])>threshold:
                    removedUser[user]=1

            for user in removedUser:
                del self.data.testSet_u[user]

            testData = []
            for item in self.data.testData:
                if item[0] not in removedUser:
                    testData.append(item)
            self.data.testData = testData

        self.num_users, self.num_items, self.train_size = self.data.trainingSize()

    def initializing_log(self):
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.log = Log(self.algorName,self.algorName+self.foldInfo+' '+currentTime)
        #save configuration
        self.log.add('### model configuration ###')
        for k in self.config.config:
            self.log.add(k+'='+self.config[k])

    def readConfiguration(self):
        self.algorName = self.config['recommender']
        self.output = LineConfig(self.config['output.setup'])
        self.isOutput = self.output.isMainOn()
        self.ranking = LineConfig(self.config['item.ranking'])

    def printAlgorConfig(self):
        "show algorithm's configuration"
        print('Algorithm:',self.config['recommender'])
        print('Ratings dataset:',abspath(self.config['ratings']))
        if LineConfig(self.config['evaluation.setup']).contains('-testSet'):
            print('Test set:',abspath(LineConfig(self.config['evaluation.setup']).getOption('-testSet')))
        #print 'Count of the users in training set: ',len()
        print('Training set size: (user count: %d, item count %d, record count: %d)' %(self.data.trainingSize()))
        print('Test set size: (user count: %d, item count %d, record count: %d)' %(self.data.testSize()))
        print('='*80)

    def initModel(self):
        pass

    def buildModel(self):
        'build the model (for model-based algorithms )'
        pass

    def buildModel_tf(self):
        'training model on tensorflow'
        pass

    def saveModel(self):
        pass

    def loadModel(self):
        pass

    def predict(self,u,i):
        pass

    def predictForRanking(self,u):
        pass


    def checkRatingBoundary(self,prediction):
        if prediction > self.data.rScale[-1]:
            return self.data.rScale[-1]
        elif prediction < self.data.rScale[0]:
            return self.data.rScale[0]
        else:
            return round(prediction,3)

    def evalRatings(self):
        res = list() #used to contain the text of the result
        res.append('userId  itemId  original  prediction\n')
        #predict
        for ind,entry in enumerate(self.data.testData):
            user,item,rating = entry
            #predict
            prediction = self.predict(user,item)
            #denormalize
            #prediction = denormalize(prediction,self.data.rScale[-1],self.data.rScale[0])
            #####################################
            pred = self.checkRatingBoundary(prediction)
            # add prediction in order to measure
            self.data.testData[ind].append(pred)
            res.append(user+' '+item+' '+str(rating)+' '+str(pred)+'\n')
        currentTime = strftime("%Y-%m-%d %H-%M-%S",localtime(time()))
        #output prediction result
        if self.isOutput:
            outDir = self.output['-dir']
            fileName = self.config['recommender']+'@'+currentTime+'-rating-predictions'+self.foldInfo+'.txt'
            FileIO.writeFile(outDir,fileName,res)
            print('The result has been output to ',abspath(outDir),'.')
        #output evaluation result
        outDir = self.output['-dir']
        fileName = self.config['recommender'] + '@'+currentTime +'-measure'+ self.foldInfo + '.txt'
        self.measure = Measure.ratingMeasure(self.data.testData)
        FileIO.writeFile(outDir, fileName, self.measure)
        self.log.add('###Evaluation Results###')
        self.log.add(self.measure)
        print('The result of %s %s:\n%s' % (self.algorName, self.foldInfo, ''.join(self.measure)))



    def evalRanking(self):
        if self.ranking.contains('-topN'):
            top = self.ranking['-topN'].split(',')
            top = [int(num) for num in top]
            N = int(top[-1])
            if N > 100 or N < 0:
                print('N can not be larger than 100! It has been reassigned with 10')
                N = 10
            if N > len(self.data.item):
                N = len(self.data.item)
        else:
            print('No correct evaluation metric is specified!')
            exit(-1)

        self.recOutput.append('userId: recommendations in (itemId, ranking score) pairs, * means the item matches.\n')
        # predict
        recList = {}
        userCount = len(self.data.testSet_u)
        #rawRes = {}
        for i, user in enumerate(self.data.testSet_u):
            itemSet = {}
            line = user + ':'
            predictedItems = self.predictForRanking(user)
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            for id, rating in enumerate(predictedItems):
                # if not self.data.rating(user, self.data.id2item[id]):
                # prediction = self.checkRatingBoundary(prediction)
                # pred = self.checkRatingBoundary(prediction)
                #####################################
                itemSet[self.data.id2item[id]] = rating
            ratedList, ratingList = self.data.userRated(user)
            for item in ratedList:
                del itemSet[item]
            recList[user] = find_k_largest(N,itemSet)
            if i % 100 == 0:
                print(self.algorName, self.foldInfo, 'progress:' + str(i) + '/' + str(userCount))
            for item in recList[user]:
                line += ' (' + item[0] + ',' + str(item[1]) + ')'
                if item[0] not in self.data.testSet_u[user]:
                    line += '*'
            line += '\n'
            self.recOutput.append(line)
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # output prediction result
        if self.isOutput:
            outDir = self.output['-dir']
            fileName = self.config['recommender'] + '@' + currentTime + '-top-' + str(
            N) + 'items' + self.foldInfo + '.txt'
            FileIO.writeFile(outDir, fileName, self.recOutput)
            print('The result has been output to ', abspath(outDir), '.')
        # output evaluation result
        outDir = self.output['-dir']
        fileName = self.config['recommender'] + '@' + currentTime + '-measure' + self.foldInfo + '.txt'
        self.measure = Measure.rankingMeasure(self.data.testSet_u, recList, top)
        self.log.add('###Evaluation Results###')
        self.log.add(self.measure)
        FileIO.writeFile(outDir, fileName, self.measure)
        print('The result of %s %s:\n%s' % (self.algorName, self.foldInfo, ''.join(self.measure)))

    def execute(self):
        self.readConfiguration()
        self.initializing_log()
        if self.foldInfo == '[1]':
            self.printAlgorConfig()
        #load model from disk or build model
        if self.isLoadModel:
            print('Loading model %s...' %self.foldInfo)
            self.loadModel()
        else:
            print('Initializing model %s...' %self.foldInfo)
            self.initModel()
            print('Building Model %s...' %self.foldInfo)
            try:
                if self.evalSettings.contains('-tf'):
                    import tensorflow
                    self.buildModel_tf()
                else:
                    self.buildModel()
            except ImportError:
                self.buildModel()

        #rating prediction or item ranking
        print('Predicting %s...' %self.foldInfo)
        if self.ranking.isMainOn():
            self.evalRanking()
        else:
            self.evalRatings()

        #save model
        if self.isSaveModel:
            print('Saving model %s...' %self.foldInfo)
            self.saveModel()
        return self.measure



