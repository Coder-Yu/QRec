#coding:utf8
from baseclass.DeepRecommender import DeepRecommender
from random import choice,shuffle
from tool.qmath import sigmoid
from math import log
import numpy as np
import random
from tool import config

try:
    import tensorflow as tf
except ImportError:
    print 'This method can only be run tensorflow!'
    exit(-1)
from tensorflow import set_random_seed
set_random_seed(2)

class APR(DeepRecommender):

    # APR：Bayesian Personalized Ranking from Implicit Feedback
    # Steffen Rendle,Christoph Freudenthaler,Zeno Gantner and Lars Schmidt-Thieme

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(APR, self).__init__(conf,trainingSet,testSet,fold)

    # def readConfiguration(self):
    #     super(APR, self).readConfiguration()

    def readConfiguration(self):
        super(APR, self).readConfiguration()
        args = config.LineConfig(self.config['APR'])
        self.eps = float(args['-eps'])
        self.regAdv = float(args['-regA'])
        self.advEpoch = int(args['-advEpoch'])


    def _create_variables(self):
        self.adv_U = tf.Variable(tf.zeros(shape=[self.m, self.k]),dtype=tf.float32, trainable=False)
        self.adv_V = tf.Variable(tf.zeros(shape=[self.n, self.k]),dtype=tf.float32, trainable=False)
        self.neg_idx = tf.placeholder(tf.int32, [None], name="n_idx")
        self.V_neg_embed = tf.nn.embedding_lookup(self.V, self.neg_idx)
        self.eps = tf.constant(self.eps,dtype=tf.float32)
        self.regAdv = tf.constant(self.regAdv,dtype=tf.float32)

    def _create_inference(self):
        result = tf.subtract(tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), 1),
                                  tf.reduce_sum(tf.multiply(self.U_embed, self.V_neg_embed), 1))
        return result

    def _create_adv_inference(self):
        self.U_plus_delta = tf.add(self.U_embed, tf.nn.embedding_lookup(self.adv_U, self.u_idx))
        self.V_plus_delta = tf.add(self.V_embed, tf.nn.embedding_lookup(self.adv_V, self.v_idx))
        self.V_neg_plus_delta = tf.add(self.V_embed, tf.nn.embedding_lookup(self.adv_V, self.neg_idx))
        result = tf.subtract(tf.reduce_sum(tf.multiply(self.U_plus_delta, self.V_plus_delta), 1),
                             tf.reduce_sum(tf.multiply(self.U_plus_delta, self.V_neg_plus_delta), 1))
        return result

    def _create_adversarial(self):
        self.grad_U, self.grad_V = tf.gradients(self.loss_adv, [self.adv_U,self.adv_V])

        # convert the IndexedSlice Data to Dense Tensor
        self.grad_U_dense = tf.stop_gradient(self.grad_U)
        self.grad_V_dense = tf.stop_gradient(self.grad_V)

        # normalization: new_grad = (grad / |grad|) * eps
        self.update_U = self.adv_U.assign(tf.nn.l2_normalize(self.grad_U_dense, 1) * self.eps)
        self.update_V = self.adv_V.assign(tf.nn.l2_normalize(self.grad_V_dense, 1) * self.eps)


    def _create_loss(self):
        self.reg_lambda = tf.constant(self.regU, dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.nn.softplus(-self._create_inference()))
        self.reg_loss = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U_embed)),
                               tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V_embed)))
        self.reg_loss = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U_embed)), self.reg_loss)
        self.total_loss = tf.add(self.loss, self.reg_loss)

        self.loss_adv = tf.multiply(self.regAdv,tf.reduce_sum(tf.nn.softplus(-self._create_adv_inference())))
        self.loss_adv = tf.add(self.loss,self.loss_adv)

    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(self.lRate)
        self.train = self.optimizer.minimize(self.total_loss)

        self.optimizer_adv = tf.train.AdamOptimizer(self.lRate)
        self.train_adv = self.optimizer.minimize(self.loss_adv)


    def initModel(self):
        super(APR, self).initModel()
        self._create_variables()
        self._create_loss()
        self._create_adversarial()
        self._create_optimizer()


    def next_batch(self):
        batch_idx = np.random.randint(self.train_size, size=self.batch_size)

        users = [self.dao.trainingData[idx][0] for idx in batch_idx]
        items = [self.dao.trainingData[idx][1] for idx in batch_idx]
        user_idx,item_idx=[],[]
        neg_item_idx = []
        for i,user in enumerate(users):
            for j in range(3): #negative count
                item_j = random.randint(0,self.n-1)

                while self.dao.trainSet_u[user].has_key(self.dao.id2item[item_j]):
                    item_j = random.randint(0, self.n - 1)

                user_idx.append(self.dao.user[user])
                item_idx.append(self.dao.item[items[i]])
                neg_item_idx.append(item_j)

        return user_idx,item_idx,neg_item_idx


    def buildModel(self):
        print 'training...'
        iteration = 0
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            # 迭代，传递变量
            for epoch in range(self.maxIter):
                # 按批优化
                user_idx,item_idx,neg_item_idx = self.next_batch()
                _,loss = sess.run([self.train,self.total_loss],feed_dict={self.u_idx: user_idx, self.v_idx: item_idx, self.neg_idx:neg_item_idx})
                print 'iteration:', epoch, 'loss:',loss

                # 输出训练完毕的矩阵
                self.P = sess.run(self.U)
                self.Q = sess.run(self.V)
                self.ranking_performance()

            for epoch in range(self.advEpoch):
                # 按批优化
                user_idx,item_idx,neg_item_idx = self.next_batch()
                sess.run([self.update_U, self.update_V],
                         feed_dict={self.u_idx: user_idx, self.v_idx: item_idx, self.neg_idx: neg_item_idx})
                _,loss = sess.run([self.train_adv,self.loss_adv],feed_dict={self.u_idx: user_idx, self.v_idx: item_idx, self.neg_idx:neg_item_idx})

                print 'iteration:', epoch, 'loss:',loss

                # 输出训练完毕的矩阵
                self.P = sess.run(self.U)
                self.Q = sess.run(self.V)
                self.ranking_performance()

    # def optimization(self,u,i,j):
    #     s = sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))
    #     self.P[u] += self.lRate * (1 - s) * (self.Q[i] - self.Q[j])
    #     self.Q[i] += self.lRate * (1 - s) * self.P[u]
    #     self.Q[j] -= self.lRate * (1 - s) * self.P[u]
    #
    #     self.P[u] -= self.lRate * self.regU * self.P[u]
    #     self.Q[i] -= self.lRate * self.regI * self.Q[i]
    #     self.Q[j] -= self.lRate * self.regI * self.Q[j]
    #     self.loss += -log(s)
    #
    # def adv_update(self,u,i,j):
    #     norm_u = l2((self.Q[i]- self.Q[j]))
    #     norm_i = l2((self.P[u]))
    #     self.adv_U[u] = self.eps*(self.Q[i]- self.Q[j])/norm_u
    #     self.adv_V[i] = self.eps*(self.P[u])/norm_i
    #     self.adv_V[j] = self.eps * (-self.P[u]) / norm_i
    #
    #     s = sigmoid((self.P[u]+self.adv_U[u]).dot(self.Q[i]+self.adv_V[i]) - (self.P[u]+self.adv_U[u]).dot(self.Q[j]+self.adv_V[j]))
    #
    #     self.P[u] += self.regAdv*self.lRate * (1 - s) * (self.Q[i] + self.adv_V[i] - self.Q[j] - self.adv_V[j])
    #     self.Q[i] += self.regAdv*self.lRate * (1 - s) * (self.P[u] + self.adv_U[u])
    #     self.Q[j] -= self.regAdv*self.lRate * (1 - s) * (self.P[u] + self.adv_U[u])
    #
    #     self.loss += -self.regAdv*log(s)

    def predict(self,user,item):

        if self.dao.containsUser(user) and self.dao.containsItem(item):
            u = self.dao.getUserId(user)
            i = self.dao.getItemId(item)
            predictRating = sigmoid(self.Q[i].dot(self.P[u]))
            return predictRating
        else:
            return sigmoid(self.dao.globalMean)

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            u = self.dao.getUserId(u)
            return self.Q.dot(self.P[u])
        else:
            return [self.dao.globalMean] * len(self.dao.item)


