# Copyright (C) 2016 School of Software Engineering, Chongqing University
#
# This file is part of RecQ.
#
# RecQ is a free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
from data.data import ratingDAO
from tool.file import FileIO
from tool.config import Config,LineConfig
from os.path import abspath
from time import strftime,localtime,time
from evaluation.measure import Measure
class Recommender(object):
    def __init__(self,configuration):
        self.config = configuration
        self.dao = None
        self.isSaveModel = False
        self.ranking = None
        self.isLoadModel = False
        self.output = None
        self.foldInfo = '[1]'
        self.isOutput = True
        self.readConfiguration()

    def readConfiguration(self):
        self.dao = ratingDAO(self.config)
        self.output = LineConfig(self.config['output.setup'])
        self.isOutput = self.output.isMainOn()
        self.ranking = LineConfig(self.config['item.ranking'])

    def printAlgorConfig(self):
        "show algorithm's configuration"
        print 'Algorithm:',self.config['recommender']
        print 'Ratings dataSet:',abspath(self.config['ratings'])
        if LineConfig(self.config['evaluation']).contains('-testSet'):
            print 'Test set:',abspath(LineConfig(self.config['evaluation']).getOption('-testSet'))
        #print 'Count of the users in training set: ',len()
        print '='*80

    def initModel(self):
        pass

    def buildModel(self):
        'build the model (for model-based algorithms )'
        pass

    def saveModel(self):
        pass

    def loadModel(self):
        pass

    def predict(self,u,i):
        pass

    def evalRatings(self):
        res = [] #used to contain the text of the result
        res.append('userId  itemId  original  prediction\n')
        #predict
        for userId in self.dao.testSet_u:
            for ind,item in enumerate(self.dao.testSet_u[userId]):
                itemId = item[0]
                originRating = item[1]
                pred = self.predict(userId,itemId)
                # add prediction in order to measure
                self.dao.testSet_u[userId][ind].append(pred)
                res.append(userId+' '+itemId+' '+str(originRating)+' '+str(pred)+'\n')
        currentTime = strftime("%Y-%m-%d %H-%M-%S",localtime(time()))
        #output prediction result
        if self.isOutput:
            outDir = self.output['-dir']
            fileName = self.config['recommender']+'@'+currentTime+'-rating-predictions'+self.foldInfo+'.txt'
            FileIO.writeFile(outDir,fileName,res)
            print 'The Result has been output to ',abspath(outDir),'.'
        #output evaluation result
        outDir = self.output['-dir']
        fileName = self.config['recommender'] + '@'+currentTime +'-measure'+ self.foldInfo + '.txt'
        measure = Measure.ratingMeasure(self.dao.testSet_u)
        FileIO.writeFile(outDir, fileName, measure)

    def evalRanking(self):
        res = []  # used to contain the text of the result
        N = self.ranking['-topN']
        res.append('userId: recommendations in (itemId, ranking score) pairs, where a correct recommendation is denoted by symbol *.')
        # predict
        topNSet = {}
        userCount = len(self.dao.testSet_u)
        for i,userId in enumerate(self.dao.testSet_u):
            itemSet = {}
            line = userId+':'
            for itemId in self.dao.item:
                pred = self.predict(userId, itemId)
                # add prediction in order to measure
                itemSet[itemId] = pred
            topNSet[userId] = sorted(itemSet.iteritems(),key=lambda d:d[1],reverse=True)[0:int(N)]

            if i%100==0:
                print 'Progress:'+str(i)+'/'+str(userCount)
            for item in topNSet[userId]:
                if self.dao.rating(userId,item[0]) != 0:
                    line+='*('+item[0]+','+str(item[1])+') '
                else:
                    line += '(' + item[0] + ',' + str(item[1]) + ') '
            line+='\n'
            res.append(line)
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # output prediction result
        if self.isOutput:
            outDir = self.output['-dir']
            fileName = self.config['recommender'] + '@' + currentTime + '-top-'+N+'items' + self.foldInfo + '.txt'
            FileIO.writeFile(outDir, fileName, res)
            print 'The Result has been output to ', abspath(outDir), '.'
        # output evaluation result
        # outDir = self.output['-dir']
        # fileName = self.config['recommender'] + '@' + currentTime + '-measure' + self.foldInfo + '.txt'
        # info = []
        # info.append('MAE ' + str(Measure.MAE(self.dao.testSet_u)))
        # FileIO.writeFile(outDir, fileName, info)

    def execute(self):
        self.printAlgorConfig()
        #load model from disk or build model
        if self.isLoadModel:
            print 'Loading model...'
            self.loadModel()
        else:
            print 'Initializing model...'
            self.initModel()
            print 'Building Model...'
            self.buildModel()

        #preict the ratings or item ranking
        print 'Predicting...'
        if self.ranking.isMainOn():
            self.evalRanking()
        else:
            self.evalRatings()

        #save model
        if self.isSaveModel:
            print 'Saving model...'
            self.saveModel()


