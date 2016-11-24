from baseclass.Recommender import Recommender
from tool import config
import numpy as np

class IterativeRecommender(Recommender):
    def __init__(self,conf):
        super(IterativeRecommender, self).__init__(conf)

    def readConfiguration(self):
        super(IterativeRecommender, self).readConfiguration()
        # set the reduced dimension
        self.k = int(self.config['num.factors'])
        # set maximum iteration
        self.maxIter = int(self.config['num.max.iter'])
        # set learning rate
        learningRate = config.LineConfig(self.config['learnRate'])
        self.lRate = float(learningRate['-init'])
        self.maxLRate = float(learningRate['-max'])
        # regularization parameter
        regular = config.LineConfig(self.config['reg.lambda'])
        self.regU,self.regI,self.regB= float(regular['-u']),float(regular['-i']),float(regular['-b'])

    def printAlgorConfig(self):
        super(IterativeRecommender, self).printAlgorConfig()
        print 'Reduced Dimension:',self.k
        print 'Maximum Iteration:',self.maxIter
        print 'Regularization parameter: regU %.3f, regI %.3f, regB %.3f' %(self.regU,self.regI,self.regB)
        print '='*80

    def initModel(self):
        self.P = np.random.rand(self.dao.trainingSize()[0], self.k)  # latent user matrix
        self.Q = np.random.rand(self.dao.trainingSize()[1], self.k)  # latent item matrix
        self.loss, self.lastLoss = 0, 0

    def saveModel(self):
        pass

    def loadModel(self):
        pass

    def updateLearningRate(self,iter):
        if iter > 1:
            if abs(self.lastLoss) > abs(self.loss):
                self.lRate *= 1.05
            else:
                self.lRate *= 0.5

        if self.maxLRate > 0 and self.lRate > self.maxLRate:
            self.lRate = self.maxLRate


    def predict(self,u,i):
        if self.dao.containsUser(u) and self.dao.containsItem(i):
            return self.P[self.dao.getUserId(u)].dot(self.Q[self.dao.getItemId(i)])
        else:
            return self.dao.globalMean

    def isConverged(self,iter):
        from math import isnan
        if isnan(self.loss):
            print 'Loss = NaN or Infinity: current settings does not fit the recommender! Change the settings and try again!'
            exit(-1)
        deltaLoss = (self.lastLoss-self.loss)
        print '%s iteration %d: loss = %.4f, delta_loss = %.4f learning_Rate = %f' %(self.algorName,iter,self.loss,deltaLoss,self.lRate)
        #check if converged
        cond = abs(deltaLoss) < 1e-3
        converged = cond
        if not converged:
            self.updateLearningRate(iter)
        self.lastLoss = self.loss
        return converged


