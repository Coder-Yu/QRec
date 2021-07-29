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
            for entry in self.data.trainingData:
                user, item, rating = entry
                u = self.data.user[user] #get user id
                i = self.data.item[item] #get item id
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

        self.r_hat = tf.reduce_sum(tf.multiply(self.user_embedding, self.item_embedding), axis=1)
        self.loss = tf.nn.l2_loss(tf.subtract(self.r, self.r_hat))
        reg_loss = self.regU*tf.nn.l2_loss(self.user_embedding) +  self.regI*tf.nn.l2_loss(self.item_embedding)
        self.total_loss = self.loss+reg_loss
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
                print('iteration:', step, 'loss:', sess.run(self.total_loss, feed_dict={self.r: rating, self.u_idx: user_idx, self.v_idx: item_idx}))

            # 输出训练完毕的矩阵
            self.P = sess.run(self.U)
            self.Q = sess.run(self.V)