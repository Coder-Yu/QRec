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
class Recommender(object):
    def __init__(self,rMatrix,configuration,data_access):
        self.ratingMatrix = rMatrix
        self.config = configuration
        self.dao = data_access
        self.isSaveModel = False
        self.ranking = False
        self.isLoadModel = False
        self.output = None
        self.foldInfo = '[1]'
        self.isOutput = True
        self.readConfiguration()



    def readConfiguration(self):
        pass

    def printAlgorConfig(self):
        "show algorithm's configuration"
        pass

    def initModel(self):
        pass

    def buildModel(self):
        'build the model (for model-based algorithms )'

    def saveModel(self):
        pass

    def loadModel(self):
        pass

    def predict(self,u,i):
        pass

    def evalRatings(self):
        res = []
        res.append('userId  itemId  original  prediction')
        #predict
        for userId in self.dao.testSet:
            for item in self.dao.testSet[userId]:
                originRating = item[0]
                itemId = item[1]
                pred = self.predict(userId,itemId)
                res.append(userId+' '+itemId+' '+originRating+' '+str(pred)+'\n')
        #output result
        if self.isOutput:
            outDir = self.output['-dir']
            fileName = self.config['recommender']+'/rating-predictions'+self.foldInfo+'.txt'
            FileIO.writeFile(outDir,fileName,res)


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
        if self.ranking:
            self.evalRatings()
        else:
            self.evalRanking()

        #save model
        if self.isSaveModel:
            self.saveModel()


