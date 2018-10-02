#coding:utf8
from baseclass.IterativeRecommender import IterativeRecommender
import numpy as np
from random import choice,random,randint
from tool import config
try:
    import tensorflow as tf
except ImportError:
    print 'This method can only be run tensorflow!'
    exit(-1)
from tensorflow import set_random_seed
set_random_seed(2)

class DMF(IterativeRecommender):

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(DMF, self).__init__(conf,trainingSet,testSet,fold)


    def next_batch(self):
        rows = np.zeros((self.batch_size,len(self.dao.item)))
        cols = np.zeros((self.batch_size,len(self.dao.user)))
        batch_idx = np.random.randint(self.train_size, size=self.batch_size)

        users = [self.dao.trainingData[idx][0] for idx in batch_idx]
        items = [self.dao.trainingData[idx][1] for idx in batch_idx]

        ratings = [self.dao.trainingData[idx][2] for idx in batch_idx]

        for user in range(users):
            uid = self.dao.user[user]
            rows[uid] = self.dao.row(user)
        for item in range(items):
            iid = self.dao.item[item]
            cols[iid] = self.dao.col(item)
        return rows,cols,ratings

    def initModel(self):
        super(DMF, self).initModel()
        n_input_u = len(self.dao.item)
        n_input_i = len(self.dao.user)
        self.negative_sp = 5
        self.n_hidden_u=[50,50]
        self.n_hidden_i=[50,50]
        self.input_u = tf.placeholder("float", [None, n_input_u])
        self.input_i = tf.placeholder("float", [None, n_input_i])


    def buildModel_tf(self):

        def init_variable(shape, name):
            return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)

        with tf.name_scope("User_Layer"):
            user_W1 = init_variable([len(self.dao.item), self.n_hidden_u[0]], "user_W1")
            self.user_out = tf.matmul(self.input_u, user_W1)
            self.regLoss = tf.nn.l2_loss(user_W1)
            for i in range(1, len(self.n_hidden_u)):
                W = init_variable([self.n_hidden_u[i-1], self.n_hidden_u[i]], "user_W" + str(i))
                b = init_variable([self.n_hidden_u[i]], "user_b" + str(i))
                self.regLoss = tf.add(self.regLoss,tf.nn.l2_loss(W))
                self.regLoss = tf.add(self.regLoss, tf.nn.l2_loss(b))
                self.user_out = tf.nn.relu(tf.add(tf.matmul(self.user_out, W), b))

        with tf.name_scope("Item_Layer"):
            item_W1 = init_variable([len(self.dao.user), self.n_hidden_i[0]], "item_W1")
            self.item_out = tf.matmul(self.input_i, item_W1)
            self.regLoss = tf.add(self.regLoss, tf.nn.l2_loss(item_W1))
            for i in range(1, len(self.n_hidden_i)):
                W = init_variable([self.input_i[i-1], self.input_i[i]], "item_W" + str(i))
                b = init_variable([self.input_i[i]], "item_b" + str(i))
                self.regLoss = tf.add(self.regLoss, tf.nn.l2_loss(W))
                self.regLoss = tf.add(self.regLoss, tf.nn.l2_loss(W))
                self.item_out = tf.nn.relu(tf.add(tf.matmul(self.item_out, W), b))

        norm_user_output = tf.sqrt(tf.reduce_sum(tf.square(self.user_out), axis=1))
        norm_item_output = tf.sqrt(tf.reduce_sum(tf.square(self.item_out), axis=1))
        self.y_ = tf.reduce_sum(tf.multiply(self.user_out, self.item_out), axis=1, keep_dims=False) / (
                norm_item_output * norm_user_output)
        self.y_ = tf.maximum(1e-6, self.y_)

        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_,labels=self.r)
        self.loss = tf.add(self.loss,self.regLoss)
        optimizer = tf.train.AdamOptimizer(self.lRate).minimize(self.loss)
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        total_batch = int(len(self.dao.trainingData)/ self.batch_size)
        for epoch in range(self.maxIter):
            for i in range(total_batch):
                users,items,ratings = self.next_batch()

                _, loss = self.sess.run([optimizer, self.loss], feed_dict={self.input_u: users,self.input_i:items,self.r:ratings})

                print self.foldInfo,"Epoch:", '%04d' % (epoch + 1),"Batch:", '%03d' %(i+1),"loss=", "{:.9f}".format(loss)
        print("Optimization Finished!")



    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            res = np.zeros(len(self.dao.item))
            input_u = np.zeros((1000,len(self.dao.item)))
            input_i = np.zeros((1000,len(self.dao.user)))
            row = self.dao.row(u)
            for i in range(1000):
                input_u[i]=row
            for i in range(len(self.dao.item)/1000):
                for n in range(1000):
                    col = self.dao.col(self.dao.id2item[i*1000+n])
                    input_i[n]=col
                res[i*1000:(i+1)*1000]=self.sess.run(self.y_, feed_dict={self.input_u: input_u, self.input_i:input_i})[0]
            remain = len(self.dao.item)-len(self.dao.item)/1000*1000
            input_u = np.zeros((remain, len(self.dao.item)))
            input_i = np.zeros((remain, len(self.dao.user)))
            for i in range(len(self.dao.item)/1000*1000,len(self.dao.iten)):
                col = self.dao.col(self.dao.id2item[len(self.dao.item)/1000*1000+i])
                input_i[i]=col
            res[len(self.dao.item)/1000*1000:len(self.dao.iten)] = self.sess.run(self.y_, feed_dict={self.input_u: input_u, self.input_i: input_i})[0]

            return res
        else:
            return [self.dao.globalMean] * len(self.dao.item)


