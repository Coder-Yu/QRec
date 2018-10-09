#coding:utf-8
from baseclass.IterativeRecommender import IterativeRecommender
import numpy as np

class PMF(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(PMF, self).__init__(conf,trainingSet,testSet,fold)

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                user, item, rating = entry
                u = self.dao.user[user] #get user id
                i = self.dao.item[item] #get item id
                error = rating - self.P[u].dot(self.Q[i])
                self.loss += error**2
                p = self.P[u]
                q = self.Q[i]

                #update latent vectors
                self.P[u] += self.lRate*(error*q-self.regU*p)
                self.Q[i] += self.lRate*(error*p-self.regI*q)

            self.loss += self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum()
            iteration += 1
            if self.isConverged(iteration):
                break

    def buildModel_tf(self):
        super(PMF, self).buildModel_tf()

        import tensorflow as tf
        # 构造损失函数 设置优化器
        reg_lambda = tf.constant(self.regU, dtype=tf.float32)
        self.r_hat = tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), axis=1)
        self.loss = tf.nn.l2_loss(tf.subtract(self.r, self.r_hat))
        self.reg_loss = tf.add(tf.multiply(reg_lambda, tf.nn.l2_loss(self.U)),
                          tf.multiply(reg_lambda, tf.nn.l2_loss(self.V)))
        self.total_loss = tf.add(self.loss, self.reg_loss)
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

                user_idx = [self.dao.user[self.dao.trainingData[idx][0]] for idx in batch_idx]
                item_idx = [self.dao.item[self.dao.trainingData[idx][1]] for idx in batch_idx]
                rating = [self.dao.trainingData[idx][2] for idx in batch_idx]
                sess.run(self.train, feed_dict={self.r: rating, self.u_idx: user_idx, self.v_idx: item_idx})
                print 'iteration:', step, 'loss:', sess.run(self.total_loss, feed_dict={self.r: rating, self.u_idx: user_idx, self.v_idx: item_idx})

            # 输出训练完毕的矩阵
            self.P = sess.run(self.U)
            self.Q = sess.run(self.V)