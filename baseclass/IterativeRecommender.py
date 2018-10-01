from baseclass.Recommender import Recommender
from tool import config
import numpy as np
from random import shuffle


class IterativeRecommender(Recommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(IterativeRecommender, self).__init__(conf,trainingSet,testSet,fold)

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
        if self.evalSettings.contains('-tf'):
            self.batch_size = int(self.config['batch_size'])
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
        self.P = np.random.rand(len(self.dao.user), self.k)/3 # latent user matrix
        self.Q = np.random.rand(len(self.dao.item), self.k)/3  # latent item matrix
        self.loss, self.lastLoss = 0, 0

    def buildModel_tf(self):
        # initialization
        import tensorflow as tf
        self.u_idx = tf.placeholder(tf.int32, [None], name="u_idx")
        self.v_idx = tf.placeholder(tf.int32, [None], name="v_idx")
        self.r = tf.placeholder(tf.float32, [None], name="rating")

        self.m, self.n, self.train_size = self.dao.trainingSize()
        self.U = tf.Variable(tf.truncated_normal(shape=[self.m, self.k], stddev=0.005), name='U')
        self.V = tf.Variable(tf.truncated_normal(shape=[self.n, self.k], stddev=0.005), name='V')

        self.U_embed = tf.nn.embedding_lookup(self.U, self.u_idx)
        self.V_embed = tf.nn.embedding_lookup(self.V, self.v_idx)


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
            return self.P[self.dao.user[u]].dot(self.Q[self.dao.item[i]])
        elif self.dao.containsUser(u) and not self.dao.containsItem(i):
            return self.dao.userMeans[u]
        elif not self.dao.containsUser(u) and self.dao.containsItem(i):
            return self.dao.itemMeans[i]
        else:
            return self.dao.globalMean


    def predictForRanking(self,u):
        'used to rank all the items for the user'
        if self.dao.containsUser(u):
            return (self.Q).dot(self.P[self.dao.user[u]])
        else:
            return [self.dao.globalMean]*len(self.dao.item)

    def isConverged(self,iter):
        from math import isnan
        if isnan(self.loss):
            print 'Loss = NaN or Infinity: current settings does not fit the recommender! Change the settings and try again!'
            exit(-1)
        measure = self.performance()
        value = [item.strip()for item in measure]
        #with open(self.algorName+' iteration.txt')
        deltaLoss = (self.lastLoss-self.loss)
        if self.ranking.isMainOn():
            print '%s %s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f' %(self.algorName,self.foldInfo,iter,self.loss,deltaLoss,self.lRate)
        else:
            print '%s %s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f %s %s' % (self.algorName, self.foldInfo, iter, self.loss, deltaLoss, self.lRate, measure[0][:11], measure[1][:12])
        #check if converged
        cond = abs(deltaLoss) < 1e-3
        converged = cond
        if not converged:
            self.updateLearningRate(iter)
        self.lastLoss = self.loss
        shuffle(self.dao.trainingData)
        return converged

