from baseclass.Recommender import Recommender
from tool import config
import numpy as np
from random import shuffle
from tool.qmath import denormalize
from evaluation.measure import Measure

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
        self.m, self.n, self.train_size = self.dao.trainingSize()

    def buildModel_tf(self):
        # initialization
        import tensorflow as tf
        self.u_idx = tf.placeholder(tf.int32, [None], name="u_idx")
        self.v_idx = tf.placeholder(tf.int32, [None], name="v_idx")
        self.r = tf.placeholder(tf.float32, [None], name="rating")

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
        deltaLoss = (self.lastLoss-self.loss)
        if self.ranking.isMainOn():
            print '%s %s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f' \
                  %(self.algorName,self.foldInfo,iter,self.loss,deltaLoss,self.lRate)
            measure = self.ranking_performance()
        else:
            measure = self.rating_performance()
            print '%s %s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f %5s %5s' \
                  % (self.algorName, self.foldInfo, iter, self.loss, deltaLoss, self.lRate, measure[0].strip()[:11], measure[1].strip()[:12])
        #check if converged
        cond = abs(deltaLoss) < 1e-3
        converged = cond
        if not converged:
            self.updateLearningRate(iter)
        self.lastLoss = self.loss
        shuffle(self.dao.trainingData)
        return converged

    def rating_performance(self):

        res = []
        for ind, entry in enumerate(self.dao.testData):
            user, item, rating = entry

            # predict
            prediction = self.predict(user, item)
            # denormalize
            prediction = denormalize(prediction, self.dao.rScale[-1], self.dao.rScale[0])
            #####################################
            pred = self.checkRatingBoundary(prediction)
            # add prediction in order to measure
            res.append([user,item,rating,pred])

        self.measure = Measure.ratingMeasure(res)

        return self.measure

    def ranking_performance(self):
        N = 10
        recList = {}
        testSample = {}
        for user in self.dao.testSet_u:
            if len(testSample) == 1000:
                break
            testSample[user] = self.dao.testSet_u[user]

        for user in testSample:
            itemSet = {}
            predictedItems = self.predictForRanking(user)
            for id, rating in enumerate(predictedItems):
                itemSet[self.dao.id2item[id]] = rating

            ratedList, ratingList = self.dao.userRated(user)
            for item in ratedList:
                del itemSet[item]

            Nrecommendations = []
            for item in itemSet:
                if len(Nrecommendations) < N:
                    Nrecommendations.append((item, itemSet[item]))
                else:
                    break

            Nrecommendations.sort(key=lambda d: d[1], reverse=True)
            recommendations = [item[1] for item in Nrecommendations]
            resNames = [item[0] for item in Nrecommendations]

            # find the K biggest scores
            for item in itemSet:
                ind = N
                l = 0
                r = N - 1

                if recommendations[r] < itemSet[item]:
                    while True:
                        mid = (l + r) / 2
                        if recommendations[mid] >= itemSet[item]:
                            l = mid + 1
                        elif recommendations[mid] < itemSet[item]:
                            r = mid - 1
                        if r < l:
                            ind = r
                            break
                # ind = bisect(recommendations, itemSet[item])
                if ind < N - 1:
                    recommendations[ind + 1] = itemSet[item]
                    resNames[ind + 1] = item
            recList[user] = zip(resNames, recommendations)
        measure = Measure.rankingMeasure(testSample, recList, [10])
        print '-'*80
        print 'Ranking Performance '+self.foldInfo+' (Top-10 On 1000 sampled users)'
        for m in measure[1:]:
            print m.strip()
        print '-'*80
        self.record.append(measure[3].strip()+' '+measure[4])
        return measure

