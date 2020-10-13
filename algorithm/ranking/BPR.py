#coding:utf8
from baseclass.IterativeRecommender import IterativeRecommender
from random import choice
from tool.qmath import sigmoid
from math import log
from collections import defaultdict
#import tensorflow as tf
class BPR(IterativeRecommender):

    # BPR：Bayesian Personalized Ranking from Implicit Feedback
    # Steffen Rendle,Christoph Freudenthaler,Zeno Gantner and Lars Schmidt-Thieme

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(BPR, self).__init__(conf,trainingSet,testSet,fold)

    # def readConfiguration(self):
    #     super(BPR, self).readConfiguration()

    def initModel(self):
        super(BPR, self).initModel()


    def buildModel(self):

        print 'Preparing item sets...'
        self.PositiveSet = defaultdict(dict)
        #self.NegativeSet = defaultdict(list)

        for user in self.data.user:
            for item in self.data.trainSet_u[user]:
                if self.data.trainSet_u[user][item] >= 1:
                    self.PositiveSet[user][item] = 1
                # else:
                #     self.NegativeSet[user].append(item)
        print 'training...'
        iteration = 0
        itemList = self.data.item.keys()
        while iteration < self.maxIter:
            self.loss = 0
            for user in self.PositiveSet:
                u = self.data.user[user]
                for item in self.PositiveSet[user]:
                    i = self.data.item[item]

                    item_j = choice(itemList)
                    while (self.PositiveSet[user].has_key(item_j)):
                        item_j = choice(itemList)
                    j = self.data.item[item_j]
                    self.optimization(u,i,j)

            self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()
            iteration += 1
            if self.isConverged(iteration):
                break

    def optimization(self,u,i,j):
        s = sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))
        self.P[u] += self.lRate * (1 - s) * (self.Q[i] - self.Q[j])
        self.Q[i] += self.lRate * (1 - s) * self.P[u]
        self.Q[j] -= self.lRate * (1 - s) * self.P[u]

        self.P[u] -= self.lRate * self.regU * self.P[u]
        self.Q[i] -= self.lRate * self.regI * self.Q[i]
        self.Q[j] -= self.lRate * self.regI * self.Q[j]
        self.loss += -log(s)

    def predict(self,user,item):

        if self.data.containsUser(user) and self.data.containsItem(item):
            u = self.data.getUserId(user)
            i = self.data.getItemId(item)
            predictRating = sigmoid(self.Q[i].dot(self.P[u]))
            return predictRating
        else:
            return sigmoid(self.data.globalMean)

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

            u_idx,i_idx,j_idx = [],[],[]
            item_list = self.data.item.keys()
            for i,user in enumerate(users):

                i_idx.append(self.data.item[items[i]])
                u_idx.append(self.data.user[user])

                neg_item = choice(item_list)
                while neg_item in self.data.trainSet_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(self.data.item[neg_item])

            yield u_idx,i_idx,j_idx

    def buildModel_tf(self):
        super(BPR, self).buildModel_tf()
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_item_embedding = tf.nn.embedding_lookup(self.V, self.neg_idx)
        y = tf.reduce_sum(tf.multiply(self.user_embedding,self.item_embedding),1)\
                                 -tf.reduce_sum(tf.multiply(self.user_embedding,self.neg_item_embedding),1)
        loss = -tf.reduce_sum(tf.log(tf.sigmoid(y))) + self.regU * (tf.nn.l2_loss(self.user_embedding) +
                                                                       tf.nn.l2_loss(self.item_embedding) +
                                                                       tf.nn.l2_loss(self.neg_item_embedding))
        opt = tf.train.AdamOptimizer(self.lRate)

        train = opt.minimize(loss)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for iteration in range(self.maxIter):
                for n,batch in enumerate(self.next_batch()):
                    user_idx, i_idx, j_idx = batch
                    _, l = sess.run([train, loss], feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx,self.v_idx: i_idx})
                    print 'training:', iteration + 1, 'batch', n, 'loss:', l
            self.P,self.Q = sess.run([self.U,self.V])



    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.Q.dot(self.P[u])
        else:
            return [self.data.globalMean] * self.num_items


