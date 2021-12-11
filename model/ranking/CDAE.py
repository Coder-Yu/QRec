#coding:utf8
from base.deepRecommender import DeepRecommender
import numpy as np
from random import choice
from util import config
import tensorflow as tf

class CDAE(DeepRecommender):

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(CDAE, self).__init__(conf,trainingSet,testSet,fold)

    def encoder(self,x,v):
        layer = tf.nn.sigmoid(tf.matmul(x, self.weights['encoder'])+self.biases['encoder']+v)
        return layer

    def decoder(self,x):
        layer = tf.nn.sigmoid(tf.matmul(x, self.weights['decoder'])+self.biases['decoder'])
        return layer

    def next_batch(self):
        X = np.zeros((self.batch_size,self.num_items))
        uids = []
        positive = np.zeros((self.batch_size, self.num_items))
        negative = np.zeros((self.batch_size, self.num_items))
        userList = list(self.data.user.keys())
        itemList = list(self.data.item.keys())
        for n in range(self.batch_size):
            user = choice(userList)
            uids.append(self.data.user[user])
            vec = self.data.row(user)
            ratedItems, values = self.data.userRated(user)
            for item in ratedItems:
                iid = self.data.item[item]
                positive[n][iid]=1
            for i in range(self.negative_sp*len(ratedItems)):
                ng = choice(itemList)
                while ng in self.data.trainSet_u:
                    ng = choice(itemList)
                n_id = self.data.item[ng]
                negative[n][n_id]=1
            X[n]=vec
        return X,uids,positive,negative

    def readConfiguration(self):
        super(CDAE, self).readConfiguration()
        args = config.OptionConf(self.config['CDAE'])
        self.corruption_level = float(args['-co'])
        self.n_hidden = int(args['-nh'])

    def initModel(self):
        super(CDAE, self).initModel()
        self.negative_sp = 5
        initializer = tf.contrib.layers.xavier_initializer()
        self.X = tf.placeholder(tf.float32, [None, self.num_items])
        self.positive = tf.placeholder(tf.float32, [None, self.num_items])
        self.negative = tf.placeholder(tf.float32, [None, self.num_items])
        self.V = tf.Variable(initializer([self.num_users, self.n_hidden]))
        self.U_embeding = tf.nn.embedding_lookup(self.V, self.u_idx)
        self.weights = {
            'encoder': tf.Variable(initializer([self.num_items, self.n_hidden])),
            'decoder': tf.Variable(initializer([self.n_hidden, self.num_items])),
        }
        self.biases = {
            'encoder': tf.Variable(initializer([self.n_hidden])),
            'decoder': tf.Variable(initializer([self.num_items])),
        }
        self.mask_corruption = tf.placeholder(tf.float32, [None, self.num_items])

    def trainModel(self):
        self.corrupted_input = tf.multiply(self.mask_corruption,self.X)
        self.encoder_op = self.encoder(self.corrupted_input,self.U_embeding)
        self.decoder_op = self.decoder(self.encoder_op)
        y_pred = tf.multiply(self.decoder_op,self.mask_corruption)
        y_pred = tf.maximum(1e-6,y_pred)
        y_positive = tf.multiply(self.positive,self.mask_corruption)
        y_negative = tf.multiply(self.negative,self.mask_corruption)
        self.loss = -tf.multiply(y_positive,tf.log(y_pred))-tf.multiply((y_negative),tf.log(1-y_pred))
        self.reg_loss = self.regU*(tf.nn.l2_loss(self.weights['encoder'])+tf.nn.l2_loss(self.weights['decoder'])+
                                   tf.nn.l2_loss(self.biases['encoder'])+tf.nn.l2_loss(self.biases['decoder']))
        self.reg_loss = self.reg_loss + self.regU*tf.nn.l2_loss(self.U_embeding)
        self.loss = tf.reduce_mean(self.loss) + self.reg_loss
        optimizer = tf.train.AdamOptimizer(self.lRate).minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)


        for epoch in range(self.maxEpoch):
            mask = np.random.binomial(1, self.corruption_level,(self.batch_size, self.num_items))
            batch_xs,users,positive,negative = self.next_batch()
            _, loss= self.sess.run([optimizer, self.loss], feed_dict={self.X: batch_xs,self.mask_corruption:mask,
                                                                                     self.u_idx:users,self.positive:positive,self.negative:negative})
            print(self.foldInfo,"Epoch:", '%04d' % (epoch + 1),"loss=", "{:.9f}".format(loss))
            #print y
            #self.ranking_performance()
        print("Optimization Finished!")



    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            vec = self.data.row(u).reshape((1,self.num_items))
            uid = [self.data.user[u]]
            return self.sess.run(self.decoder_op,feed_dict={self.X:vec,self.mask_corruption:np.ones((1,self.num_items)),self.u_idx:uid})[0]
        else:
            return [self.data.globalMean] * self.num_items


