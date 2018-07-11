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
            for entry in self.dao.trainingData:
                user, item, rating = entry
                u = self.dao.user[user]
                i = self.dao.item[item]
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
        import tensorflow as tf
        #占位符，存储输入的用户id和商品id以及评分
        u_idx = tf.placeholder(tf.int32, [None], name="u_idx")
        v_idx = tf.placeholder(tf.int32, [None], name="v_idx")
        r = tf.placeholder(tf.float32, [None], name="rating")
        #变量，初始化用户矩阵和商品矩阵
        m,n,train_size=self.dao.trainingSize()
        self.U = tf.Variable(tf.truncated_normal(shape=[m, self.k], stddev=0.001), name='U')
        self.V = tf.Variable(tf.truncated_normal(shape=[n, self.k], stddev=0.001), name='V')

        #取出对应的用户和商品列
        U_embed = tf.nn.embedding_lookup(self.U, u_idx)
        V_embed = tf.nn.embedding_lookup(self.V, v_idx)

        #构造损失函数 设置优化器
        r_hat = tf.reduce_sum(tf.multiply(U_embed, V_embed), reduction_indices=1)
        loss =  tf.nn.l2_loss(tf.subtract(r, r_hat))
        optimizer = tf.train.AdamOptimizer(self.lRate)
        train = optimizer.minimize(loss,var_list=[self.U,self.V])

        #初始化会话
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            #迭代，传递变量
            for step in range(self.maxIter):
                print 'iteration:',step
                #按批优化
                batch_size=self.batch_size

                batch_idx = np.random.randint(train_size, size=batch_size)

                user_idx = [self.dao.user[self.dao.trainingData[idx][0]] for idx in batch_idx]
                item_idx = [self.dao.item[self.dao.trainingData[idx][1]] for idx in batch_idx]
                rating = [self.dao.trainingData[idx][2] for idx in batch_idx]
                sess.run(train,feed_dict={r:rating,u_idx:user_idx,v_idx:item_idx})


            #输出训练完毕的矩阵
            self.P =sess.run(self.U)
            self.Q =sess.run(self.V)


