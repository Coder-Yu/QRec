#coding:utf-8
from baseclass.IterativeRecommender import IterativeRecommender
import numpy as np

class BasicMF(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(BasicMF, self).__init__(conf,trainingSet,testSet,fold)

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
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


            iteration += 1
            if self.isConverged(iteration):
                break

    def buildModel_tf(self):
        super(BasicMF, self).buildModel_tf()

        import tensorflow as tf
        # 构造损失函数 设置优化器

        self.r_hat = tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), axis=1)
        self.total_loss = tf.nn.l2_loss(self.r- self.r_hat)

        self.optimizer = tf.train.AdamOptimizer(self.lRate)
        self.train = self.optimizer.minimize(self.total_loss, var_list=[self.U, self.V])

        # 初始化会话
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            # 迭代，传递变量
            for step in range(self.maxIter):
                # 按批优化
                batch_size = self.batch_size

                batch_idx = np.random.randint(self.train_size, size=batch_size)

                user_idx = [self.data.user[self.data.trainingData[idx][0]] for idx in batch_idx]
                item_idx = [self.data.item[self.data.trainingData[idx][1]] for idx in batch_idx]
                rating = [self.data.trainingData[idx][2] for idx in batch_idx]
                sess.run(self.train, feed_dict={self.r: rating, self.u_idx: user_idx, self.v_idx: item_idx})
                print 'iteration:', step, 'loss:', sess.run(self.total_loss,
                                                            feed_dict={self.r: rating, self.u_idx: user_idx,
                                                                       self.v_idx: item_idx})

            # 输出训练完毕的矩阵
            self.P = sess.run(self.U)
            self.Q = sess.run(self.V)

