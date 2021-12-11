#coding:utf-8
from base.iterativeRecommender import IterativeRecommender
import numpy as np

class BasicMF(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(BasicMF, self).__init__(conf,trainingSet,testSet,fold)

    def trainModel(self):
        epoch = 0
        while epoch < self.maxEpoch:
            self.loss = 0
            for entry in self.data.trainingData:
                user, item, rating = entry
                u = self.data.user[user]
                i = self.data.item[item]
                error = rating - self.P[u].dot(self.Q[i])
                self.loss += error**2
                p = self.P[u]
                q = self.Q[i]
                #update latent vectors
                self.P[u] += self.lRate*error*q
                self.Q[i] += self.lRate*error*p
            epoch += 1
            if self.isConverged(epoch):
                break

    def trainModel_tf(self):
        super(BasicMF, self).trainModel_tf()
        import tensorflow as tf
        self.r_hat = tf.reduce_sum(tf.multiply(self.user_embedding, self.item_embedding), axis=1)
        self.total_loss = tf.nn.l2_loss(self.r- self.r_hat)
        self.optimizer = tf.train.AdamOptimizer(self.lRate)
        self.train = self.optimizer.minimize(self.total_loss, var_list=[self.U, self.V])

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for step in range(self.maxEpoch):
                batch_size = self.batch_size
                batch_idx = np.random.randint(self.train_size, size=batch_size)
                user_idx = [self.data.user[self.data.trainingData[idx][0]] for idx in batch_idx]
                item_idx = [self.data.item[self.data.trainingData[idx][1]] for idx in batch_idx]
                rating = [self.data.trainingData[idx][2] for idx in batch_idx]
                sess.run(self.train, feed_dict={self.r: rating, self.u_idx: user_idx, self.v_idx: item_idx})
                print('epoch:', step, 'loss:', sess.run(self.total_loss,
                                                            feed_dict={self.r: rating, self.u_idx: user_idx,
                                                                       self.v_idx: item_idx}))
            self.P = sess.run(self.U)
            self.Q = sess.run(self.V)

