from baseclass.Recommender import Recommender
from tool import config
import numpy as np
from random import shuffle
from evaluation.measure import Measure

class IterativeRecommender(Recommender):
    def __init__(self,conf,trainingSet,testSet,fold='[1]'):
        super(IterativeRecommender, self).__init__(conf,trainingSet,testSet,fold)
        self.bestPerformance = []

    def readConfiguration(self):
        super(IterativeRecommender, self).readConfiguration()
        # set the reduced dimension
        self.embed_size = int(self.config['num.factors'])
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
        print 'Reduced Dimension:',self.embed_size
        print 'Maximum Iteration:',self.maxIter
        print 'Regularization parameter: regU %.3f, regI %.3f, regB %.3f' %(self.regU,self.regI,self.regB)
        print '='*80

    def initModel(self):
        self.P = np.random.rand(len(self.data.user), self.embed_size)/3 # latent user matrix
        self.Q = np.random.rand(len(self.data.item), self.embed_size)/3  # latent item matrix
        self.loss, self.lastLoss = 0, 0

    def buildModel_tf(self):
        # initialization
        import tensorflow as tf
        self.u_idx = tf.placeholder(tf.int32, [None], name="u_idx")
        self.v_idx = tf.placeholder(tf.int32, [None], name="v_idx")
        self.r = tf.placeholder(tf.float32, [None], name="rating")

        self.U = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embed_size], stddev=0.005), name='U')
        self.V = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embed_size], stddev=0.005), name='V')

        self.user_biases = tf.Variable(tf.truncated_normal(shape=[self.num_users, 1], stddev=0.005), name='U')
        self.item_biases = tf.Variable(tf.truncated_normal(shape=[self.num_items, 1], stddev=0.005), name='U')

        self.user_bias = tf.nn.embedding_lookup(self.user_biases, self.u_idx)
        self.item_bias = tf.nn.embedding_lookup(self.item_biases, self.v_idx)

        self.user_embedding = tf.nn.embedding_lookup(self.U, self.u_idx)
        self.item_embedding = tf.nn.embedding_lookup(self.V, self.v_idx)

    def updateLearningRate(self,iter):
        if iter > 1:
            if abs(self.lastLoss) > abs(self.loss):
                self.lRate *= 1.05
            else:
                self.lRate *= 0.5
        if self.lRate > self.maxLRate > 0:
            self.lRate = self.maxLRate

    def predict(self,u,i):
        if self.data.containsUser(u) and self.data.containsItem(i):
            return self.P[self.data.user[u]].dot(self.Q[self.data.item[i]])
        elif self.data.containsUser(u) and not self.data.containsItem(i):
            return self.data.userMeans[u]
        elif not self.data.containsUser(u) and self.data.containsItem(i):
            return self.data.itemMeans[i]
        else:
            return self.data.globalMean

    def predictForRanking(self,u):
        'used to rank all the items for the user'
        if self.data.containsUser(u):
            return self.Q.dot(self.P[self.data.user[u]])
        else:
            return [self.data.globalMean]*self.num_items

    def isConverged(self,iter):
        from math import isnan
        if isnan(self.loss):
            print 'Loss = NaN or Infinity: current settings does not fit the recommender! Change the settings and try again!'
            exit(-1)
        deltaLoss = (self.lastLoss-self.loss)
        if self.ranking.isMainOn():
            print '%s %s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f' \
                  %(self.algorName,self.foldInfo,iter,self.loss,deltaLoss,self.lRate)
            measure = self.ranking_performance(iter)
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
        shuffle(self.data.trainingData)
        return converged

    def rating_performance(self):
        res = []
        for ind, entry in enumerate(self.data.testData):
            user, item, rating = entry
            # predict
            prediction = self.predict(user, item)
            pred = self.checkRatingBoundary(prediction)
            res.append([user,item,rating,pred])
        self.measure = Measure.ratingMeasure(res)
        return self.measure

    def ranking_performance(self,iteration):
        #for quick evaluation, we only rank 2000 items
        #results of 1000 users would be evaluated
        N = 10
        recList = {}
        testSample = {}
        for user in self.data.testSet_u:
            if len(testSample) == 1000:
                break
            if user not in self.data.trainSet_u:
                continue
            testSample[user] = self.data.testSet_u[user]

        for user in testSample:
            itemSet = {}
            predictedItems = self.predictForRanking(user)
            for id, rating in enumerate(predictedItems[:2000]):
                itemSet[self.data.id2item[id]] = rating
            ratedList, ratingList = self.data.userRated(user)
            for item in ratedList:
                if item in itemSet:
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
                if ind < N - 2:
                    recommendations[ind + 2:] = recommendations[ind + 1:-1]
                    resNames[ind + 2:] = resNames[ind + 1:-1]
                if ind < N - 1:
                    recommendations[ind + 1] = itemSet[item]
                    resNames[ind + 1] = item
            recList[user] = zip(resNames, recommendations)
        measure = Measure.rankingMeasure(testSample, recList, [10])
        if len(self.bestPerformance)>0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k,v = m.strip().split(':')
                performance[k]=float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -=1
            if count<0:
                self.bestPerformance[1]=performance
                self.bestPerformance[0]=iteration
                self.saveModel()
        else:
            self.bestPerformance.append(iteration)
            performance = {}
            for m in measure[1:]:
                k,v = m.strip().split(':')
                performance[k]=float(v)
                self.bestPerformance.append(performance)
            self.saveModel()
        print '-'*120
        print 'Quick Ranking Performance '+self.foldInfo+' (Top-10 Item Recommendation On 1000 sampled users)'
        measure = [m.strip() for m in measure[1:]]
        print '*Current Performance*'
        print 'iteration:',iteration,' | '.join(measure)
        bp = ''
        # for k in self.bestPerformance[1]:
        #     bp+=k+':'+str(self.bestPerformance[1][k])+' | '
        bp += 'Precision'+':'+str(self.bestPerformance[1]['Precision'])+' | '
        bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + ' | '
        bp += 'F1' + ':' + str(self.bestPerformance[1]['F1']) + ' | '
        bp += 'MAP' + ':' + str(self.bestPerformance[1]['MAP']) + ' | '
        bp += 'MDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        print '*Best Performance* '
        print 'iteration:',self.bestPerformance[0],bp
        print '-'*120
        return measure

