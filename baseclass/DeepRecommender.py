from baseclass.IterativeRecommender import IterativeRecommender
from random import shuffle,randint,choice
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

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def next_batch_pairwise(self):
        batch_id = 0
        while batch_id < self.train_size:
            if batch_id + self.batch_size <= self.train_size:
                users = [self.data.trainingData[idx][0] for idx in range(batch_id, self.batch_size + batch_id)]
                items = [self.data.trainingData[idx][1] for idx in range(batch_id, self.batch_size + batch_id)]
                batch_id += self.batch_size
            else:
                users = [self.data.trainingData[idx][0] for idx in range(batch_id, self.train_size)]
                items = [self.data.trainingData[idx][1] for idx in range(batch_id, self.train_size)]
                batch_id = self.train_size

            u_idx, i_idx, j_idx = [], [], []
            item_list = self.data.item.keys()
            for i, user in enumerate(users):

                i_idx.append(self.data.item[items[i]])
                u_idx.append(self.data.user[user])

                neg_item = choice(item_list)
                while neg_item in self.data.trainSet_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(self.data.item[neg_item])

            yield u_idx, i_idx, j_idx

    def next_batch_pointwise(self):
        batch_id=0
        while batch_id<self.train_size:
            if batch_id+self.batch_size<=self.train_size:
                users = [self.data.trainingData[idx][0] for idx in range(batch_id,self.batch_size+batch_id)]
                items = [self.data.trainingData[idx][1] for idx in range(batch_id,self.batch_size+batch_id)]
                batch_id+=self.batch_size
            else:
                users = [self.data.trainingData[idx][0] for idx in range(batch_id, self.train_size)]
                items = [self.data.trainingData[idx][1] for idx in range(batch_id, self.train_size)]
                batch_id=self.train_size
            u_idx,i_idx,y = [],[],[]
            for i,user in enumerate(users):
                i_idx.append(self.data.item[items[i]])
                u_idx.append(self.data.user[user])
                y.append(1)
                for instance in range(4):
                    item_j = randint(0, self.num_items - 1)
                    while self.data.trainSet_u[user].has_key(self.data.id2item[item_j]):
                        item_j = randint(0, self.num_items - 1)
                    u_idx.append(self.data.user[user])
                    i_idx.append(item_j)
                    y.append(0)
            yield u_idx,i_idx,y

    def predictForRanking(self,u):
        'used to rank all the items for the user'
        pass

    # def isConverged(self,iter):
    #     from math import isnan
    #     if isnan(self.loss):
    #         print 'Loss = NaN or Infinity: current settings does not fit the recommender! Change the settings and try again!'
    #         exit(-1)
    #     deltaLoss = (self.lastLoss-self.loss)
    #     if self.ranking.isMainOn():
    #         measure = self.ranking_performance(iter)
    #         print '%s %s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f %s %s (Top-10 On 300 users)' \
    #               %(self.algorName,self.foldInfo,iter,self.loss,deltaLoss,self.lRate, measure[-3].strip()[:11], measure[-2].strip()[:12])
    #     else:
    #         measure = self.rating_performance()
    #         print '%s %s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f %5s %5s' \
    #               % (self.algorName, self.foldInfo, iter, self.loss, deltaLoss, self.lRate, measure[0].strip()[:11], measure[1].strip()[:12])
    #     #check if converged
    #     cond = abs(deltaLoss) < 1e-6
    #     converged = cond
    #     if not converged:
    #         self.updateLearningRate(iter)
    #     self.lastLoss = self.loss
    #     shuffle(self.data.trainingData)
    #     return converged

