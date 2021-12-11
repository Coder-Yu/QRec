from base.socialRecommender import SocialRecommender
from math import log
import numpy as np
import tensorflow as tf
from util.qmath import sigmoid
from random import choice
from collections import defaultdict
class SBPR(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=None,fold='[1]'):
        super(SBPR, self).__init__(conf,trainingSet,testSet,relation,fold)

    def initModel(self):
        super(SBPR, self).initModel()
        print('Preparing item sets...')
        self.PositiveSet = defaultdict(dict)
        self.FPSet = defaultdict(dict)
        for user in self.data.user:
            for item in self.data.trainSet_u[user]:
                if self.data.trainSet_u[user][item] >= 1:
                    self.PositiveSet[user][item] = 1
            if user in self.social.user:
                for friend in self.social.getFollowees(user):
                    if friend in self.data.user:
                        for item in self.data.trainSet_u[friend]:
                            if item not in self.PositiveSet[user]:
                                if item not in self.FPSet[user]:
                                    self.FPSet[user][item] = 1
                                else:
                                    self.FPSet[user][item] += 1

    def trainModel(self):
        self.b = np.random.random(self.num_items)
        print('Training...')
        epoch = 0
        while epoch < self.maxEpoch:
            self.loss = 0
            itemList = list(self.data.item.keys())
            for user in self.PositiveSet:
                u = self.data.user[user]
                kItems = list(self.FPSet[user].keys())
                for item in self.PositiveSet[user]:
                    i = self.data.item[item]
                    if len(self.FPSet[user]) > 0:
                        item_k = choice(kItems)
                        k = self.data.item[item_k]
                        Suk = self.FPSet[user][kItems]
                        s = sigmoid((self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[k])+self.b[i]-self.b[k])/ (Suk+1))
                        self.P[u] += 1 / (Suk+1) *self.lRate * (1 - s) * (self.Q[i] - self.Q[k])
                        self.Q[i] += 1 / (Suk+1) *self.lRate * (1 - s) * self.P[u]
                        self.Q[k] -= 1 / (Suk+1) *self.lRate * (1 - s) * self.P[u]
                        item_j = choice(itemList)
                        while item_j in self.PositiveSet[user] or item_j in self.FPSet:
                            item_j = choice(itemList)
                        j = self.data.item[item_j]
                        s = sigmoid(self.P[u].dot(self.Q[k]) - self.P[u].dot(self.Q[j])+self.b[k]-self.b[j])
                        self.P[u] +=  self.lRate * (1 - s) * (self.Q[k] - self.Q[j])
                        self.Q[k] += self.lRate * (1 - s) * self.P[u]
                        self.Q[j] -= self.lRate * (1 - s) * self.P[u]
                        self.P[u] -= self.lRate * self.regU * self.P[u]
                        self.Q[i] -= self.lRate * self.regI * self.Q[i]
                        self.Q[j] -= self.lRate * self.regI * self.Q[j]
                        self.Q[k] -= self.lRate * self.regI * self.Q[k]
                        self.loss += -log(sigmoid((self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[k]))/ (Suk+1))) \
                                     - log(sigmoid(self.P[u].dot(self.Q[k]) - self.P[u].dot(self.Q[j])))
                    else:
                        item_j = choice(itemList)
                        while item_j in self.PositiveSet[user]:
                            item_j = choice(itemList)
                        j = self.data.item[item_j]
                        s = sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j])+self.b[i]-self.b[j])
                        self.P[u] += self.lRate * (1 - s) * (self.Q[i] - self.Q[j])
                        self.Q[i] += self.lRate * (1 - s) * self.P[u]
                        self.Q[j] -= self.lRate * (1 - s) * self.P[u]
                        self.loss += -log(s)
                self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()+self.b.dot(self.b)
            epoch += 1
            if self.isConverged(epoch):
                break


    def next_batch(self):
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

            u_idx,i_idx,f_idx,j_idx,weights = [],[],[],[],[]
            item_list = list(self.data.item.keys())
            for i,user in enumerate(users):
                i_idx.append(self.data.item[items[i]])
                u_idx.append(self.data.user[user])
                if len(self.FPSet[user])==0:
                    f_item = choice(item_list)
                    weights.append(0)
                else:
                    f_item = choice(list(self.FPSet[user].keys()))
                    weights.append(self.FPSet[user][f_item])
                f_idx.append(self.data.item[f_item])
                neg_item = choice(item_list)
                while neg_item in self.data.trainSet_u[user] or neg_item in self.FPSet[user]:
                    neg_item = choice(item_list)
                j_idx.append(self.data.item[neg_item])
            yield u_idx,i_idx,f_idx,j_idx,weights

    def trainModel_tf(self):
        super(SBPR, self).trainModel_tf()
        self.social_idx = tf.placeholder(tf.int32, name="social_holder")
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.weights = tf.placeholder(tf.float32, name="weights")
        self.neg_item_embedding = tf.nn.embedding_lookup(self.V, self.neg_idx)
        self.social_item_embedding = tf.nn.embedding_lookup(self.V, self.social_idx)
        y_ik = (tf.reduce_sum(tf.multiply(self.user_embedding, self.item_embedding), 1)
                -tf.reduce_sum(tf.multiply(self.user_embedding, self.social_item_embedding), 1))/(self.weights+1)
        y_kj = tf.reduce_sum(tf.multiply(self.user_embedding, self.social_item_embedding), 1)\
               -tf.reduce_sum(tf.multiply(self.user_embedding, self.neg_item_embedding), 1)
        loss = -tf.reduce_sum(tf.log(tf.sigmoid(y_ik)+1e-6)+ tf.log(tf.sigmoid(y_kj)+1e-6))
        + self.regU * (tf.nn.l2_loss(self.U) + tf.nn.l2_loss(self.V))
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(self.maxEpoch):
                for n, batch in enumerate(self.next_batch()):
                    user_idx, i_idx, s_idx,j_idx,weights = batch
                    _, l = sess.run([train, loss],
                                    feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx,self.social_idx:s_idx,self.weights:weights})
                    print('training:', epoch + 1, 'batch', n, 'loss:', l)
            self.P, self.Q = sess.run([self.U, self.V])

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.Q.dot(self.P[u])
        else:
            return [self.data.globalMean] * self.num_items


