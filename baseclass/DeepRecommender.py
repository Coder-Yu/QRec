from baseclass.IterativeRecommender import IterativeRecommender
from tool import config
import numpy as np
from random import shuffle
import tensorflow as tf

class DeepRecommender(IterativeRecommender):
    def __init__(self,conf,trainingSet,testSet,fold='[1]'):
        super(DeepRecommender, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(DeepRecommender, self).readConfiguration()
        # set the reduced dimension
        self.batch_size = int(self.config['batch_size'])


    def printAlgorConfig(self):
        super(DeepRecommender, self).printAlgorConfig()


    def initModel(self):
        super(DeepRecommender, self).initModel()
        self.u_idx = tf.placeholder(tf.int32, name="u_idx")
        self.v_idx = tf.placeholder(tf.int32, name="v_idx")

        self.r = tf.placeholder(tf.float32, name="rating")

        self.user_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embed_size], stddev=0.005), name='U')
        self.item_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embed_size], stddev=0.005), name='V')

        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.v_idx)
        self.sess = tf.Session()



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
            print '%s %s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f %s %s (Top-10 On 300 users)' \
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
        shuffle(self.data.trainingData)
        return converged

