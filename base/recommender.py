# QRec is a free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
from data.rating import Rating
from util.io import FileIO
from util.config import OptionConf
from util.log import Log
from os.path import abspath
from time import strftime,localtime,time
from util.measure import Measure
from util.qmath import find_k_largest

class Recommender(object):
    def __init__(self,conf,trainingSet,testSet,fold='[1]'):
        self.config = conf
        self.data = None
        self.isSaveModel = False
        self.ranking = None
        self.isLoadModel = False
        self.output = None
        self.isOutput = True
        self.data = Rating(self.config, trainingSet, testSet)
        self.foldInfo = fold
        self.evalSettings = OptionConf(self.config['evaluation.setup'])
        self.measure = []
        self.recOutput = []
        self.num_users, self.num_items, self.train_size = self.data.trainingSize()

    def initializing_log(self):
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.log = Log(self.modelName, self.modelName + self.foldInfo + ' ' + currentTime)
        #save configuration
        self.log.add('### model configuration ###')
        for k in self.config.config:
            self.log.add(k+'='+self.config[k])

    def readConfiguration(self):
        self.modelName = self.config['model.name']
        self.output = OptionConf(self.config['output.setup'])
        self.isOutput = self.output.isMainOn()
        self.ranking = OptionConf(self.config['item.ranking'])

    def printAlgorConfig(self):
        "show model's configuration"
        print('Model:',self.config['model.name'])
        print('Ratings dataset:',abspath(self.config['ratings']))
        if OptionConf(self.config['evaluation.setup']).contains('-testSet'):
            print('Test set:', abspath(OptionConf(self.config['evaluation.setup'])['-testSet']))
        #print dataset statistics
        print('Training set size: (user count: %d, item count %d, record count: %d)' %(self.data.trainingSize()))
        print('Test set size: (user count: %d, item count %d, record count: %d)' %(self.data.testSize()))
        print('='*80)
        #print specific parameters if applicable
        if self.config.contains(self.config['model.name']):
            parStr = ''
            args = OptionConf(self.config[self.config['model.name']])
            for key in args.keys():
                parStr+=key[1:]+':'+args[key]+'  '
            print('Specific parameters:',parStr)
            print('=' * 80)

    def initModel(self):
        pass

    def trainModel(self):
        'build the model (for model-based Models )'
        pass

    def trainModel_tf(self):
        'training model on tensorflow'
        pass

    def saveModel(self):
        pass

    def loadModel(self):
        pass

    #for rating prediction
    def predictForRating(self, u, i):
        pass

    #for item prediction
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
            prediction = self.predictForRating(user, item)
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
            fileName = self.config['model.name']+'@'+currentTime+'-rating-predictions'+self.foldInfo+'.txt'
            FileIO.writeFile(outDir,fileName,res)
            print('The result has been output to ',abspath(outDir),'.')
        #output evaluation result
        outDir = self.output['-dir']
        fileName = self.config['model.name'] + '@'+currentTime +'-measure'+ self.foldInfo + '.txt'
        self.measure = Measure.ratingMeasure(self.data.testData)
        FileIO.writeFile(outDir, fileName, self.measure)
        self.log.add('###Evaluation Results###')
        self.log.add(self.measure)
        print('The result of %s %s:\n%s' % (self.modelName, self.foldInfo, ''.join(self.measure)))

    def evalRanking(self):
        if self.ranking.contains('-topN'):
            top = self.ranking['-topN'].split(',')
            top = [int(num) for num in top]
            N = max(top)
            if N > 100 or N < 1:
                print('N can not be larger than 100! It has been reassigned to 10')
                N = 10
        else:
            print('No correct evaluation metric is specified!')
            exit(-1)
        self.recOutput.append('userId: recommendations in (itemId, ranking score) pairs, * means the item matches.\n')
        # predict
        recList = {}
        userCount = len(self.data.testSet_u)
        #rawRes = {}
        for i, user in enumerate(self.data.testSet_u):
            line = user + ':'
            candidates = self.predictForRanking(user)
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            ratedList, ratingList = self.data.userRated(user)
            for item in ratedList:
                candidates[self.data.item[item]] = 0
            ids,scores = find_k_largest(N,candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            recList[user] = list(zip(item_names,scores))
            if i % 100 == 0:
                print(self.modelName, self.foldInfo, 'progress:' + str(i) + '/' + str(userCount))
            for item in recList[user]:
                line += ' (' + item[0] + ',' + str(item[1]) + ')'
                if item[0] in self.data.testSet_u[user]:
                    line += '*'
            line += '\n'
            self.recOutput.append(line)
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # output prediction result
        if self.isOutput:
            outDir = self.output['-dir']
            fileName = self.config['model.name'] + '@' + currentTime + '-top-' + str(
            N) + 'items' + self.foldInfo + '.txt'
            FileIO.writeFile(outDir, fileName, self.recOutput)
            print('The result has been output to ', abspath(outDir), '.')
        # output evaluation result
        if self.evalSettings.contains('-predict'):
            #no evalutation
            exit(0)
        outDir = self.output['-dir']
        fileName = self.config['model.name'] + '@' + currentTime + '-measure' + self.foldInfo + '.txt'
        self.measure = Measure.rankingMeasure(self.data.testSet_u, recList, top)
        self.log.add('###Evaluation Results###')
        self.log.add(self.measure)
        FileIO.writeFile(outDir, fileName, self.measure)
        print('The result of %s %s:\n%s' % (self.modelName, self.foldInfo, ''.join(self.measure)))

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
                    self.trainModel_tf()
                else:
                    self.trainModel()
            except ImportError:
                self.trainModel()
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



