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
        print '='*50

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
        res = []
        res.append('userId  itemId  original  prediction\n')
        #predict
        for userId in self.dao.testSet:
            for item in self.dao.testSet[userId]:
                originRating = item[0]
                itemId = item[1]
                pred = self.predict(userId,itemId)
                res.append(userId+' '+itemId+' '+str(originRating)+' '+str(pred)+'\n')
        #output result
        if self.isOutput:
            outDir = self.output['-dir']
            fileName = self.config['recommender']+'-rating-predictions'+self.foldInfo+'.txt'
            FileIO.writeFile(outDir,fileName,res)
            print 'The Result has been output to ',abspath(outDir),'.'


    def evalRanking(self):
        pass

    def execute(self):
        self.printAlgorConfig()
        #load model from disk or build model
        if self.isLoadModel:
            self.loadModel()
        else:
            self.initModel()
            self.buildModel()

        #preict the ratings or item ranking
        if self.ranking.isMainOn():
            self.evalRanking()
        else:
            self.evalRatings()

        #save model
        if self.isSaveModel:
            self.saveModel()


