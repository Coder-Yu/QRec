from baseclass.IterativeRecommender import IterativeRecommender
from tool import config
import numpy as np
from random import shuffle


class DeepRecommender(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(DeepRecommender, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(DeepRecommender, self).readConfiguration()
        # set the reduced dimension
        self.batch_size = int(self.config['batch_size'])
        # regularization parameter
        regular = config.LineConfig(self.config['reg.lambda'])
        self.regU,self.regI,self.regB= float(regular['-u']),float(regular['-i']),float(regular['-b'])

    def printAlgorConfig(self):
        super(DeepRecommender, self).printAlgorConfig()


    def initModel(self):
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

    def predictForRanking(self,u):
        'used to rank all the items for the user'
        pass

    def isConverged(self,iter):
        from math import isnan
        if isnan(self.loss):
            print 'Loss = NaN or Infinity: current settings does not fit the recommender! Change the settings and try again!'
            exit(-1)
        deltaLoss = (self.lastLoss-self.loss)
        if self.ranking.isMainOn():
            measure = self.ranking_performance()
            print '%s %s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f %s %s (Top-10 On 200 users)' \
                  %(self.algorName,self.foldInfo,iter,self.loss,deltaLoss,self.lRate, measure[-3].strip()[:11], measure[-2].strip()[:12])
        else:
            measure = self.rating_performance()
            print '%s %s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f %5s %5s' \
                  % (self.algorName, self.foldInfo, iter, self.loss, deltaLoss, self.lRate, measure[0].strip()[:11], measure[1].strip()[:12])
        #check if converged
        cond = abs(deltaLoss) < 1e-8
        converged = cond
        if not converged:
            self.updateLearningRate(iter)
        self.lastLoss = self.loss
        shuffle(self.dao.trainingData)
        return converged

    def ranking_performance(self):
        measure = super(DeepRecommender, self).ranking_performance()
        print '-'*80
        print self.foldInfo,'Ranking Performance: %s %s (Top-10 On 500 sampled users)' %(measure[-3].strip()[:11], measure[-2].strip()[:12])
        print '-'*80