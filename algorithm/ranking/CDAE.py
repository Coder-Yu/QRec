#coding:utf8
from baseclass.IterativeRecommender import IterativeRecommender
import numpy as np
from random import choice,random

try:
    import tensorflow as tf
except ImportError:
    print 'This method can only be run tensorflow!'
    exit(-1)

class CDAE(IterativeRecommender):

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(CDAE, self).__init__(conf,trainingSet,testSet,fold)

    def encoder(self,x,v):
        layer = tf.nn.sigmoid(tf.add(tf.add(tf.matmul(x, self.weights['encoder']), self.biases['encoder']),v))
        return layer

    def decoder(self,x):
        layer = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder']),self.biases['decoder']))
        return layer

    def next_batch(self):
        X = np.zeros((self.batch_size,len(self.dao.item)))
        uids = []
        evaluated = np.zeros((self.batch_size,len(self.dao.item)))>0
        userList = self.dao.user.keys()
        itemList = self.dao.item.keys()
        for n in range(self.batch_size):
            sample = []
            user = choice(userList)
            uids.append(self.dao.user[user])
            vec = self.dao.row(user)
            #corrupt
            ratedItems, values = self.dao.userRated(user)
            for item in ratedItems:
                iid = self.dao.item[item]
                if random()>0.9:
                    vec[iid]=0
                evaluated[n][iid]=True
            for i in range(self.negative_sp*len(ratedItems)):
                ng = choice(itemList)
                while self.dao.trainSet_u.has_key(ng):
                    ng = choice(itemList)
                ng = self.dao.item[ng]
                evaluated[n][ng]=True
            X[n]=vec
        return X,uids,evaluated



    def initModel(self):
        super(CDAE, self).initModel()
        n_input = len(self.dao.item)
        n_hidden = 128
        n_output = len(self.dao.item)
        self.negative_sp = 5
        self.X = tf.placeholder("float", [None, n_input])
        self.sample = tf.placeholder("bool", [None, n_input])
        self.zeros = np.zeros((self.batch_size,n_input))
        self.V = tf.Variable(tf.random_normal([len(self.dao.user), n_hidden]))
        self.v_idx = tf.placeholder(tf.int32, [None], name="v_idx")
        self.V_embed = tf.nn.embedding_lookup(self.V, self.v_idx)


        self.weights = {
            'encoder': tf.Variable(tf.random_normal([n_input, n_hidden])),
            'decoder': tf.Variable(tf.random_normal([n_hidden, n_output])),
        }
        self.biases = {
            'encoder': tf.Variable(tf.random_normal([n_hidden])),
            'decoder': tf.Variable(tf.random_normal([n_output])),
        }


    def buildModel_tf(self):

        self.encoder_op = self.encoder(self.X,self.V_embed)
        self.decoder_op = self.decoder(self.encoder_op)


        y_pred = tf.where(self.sample,self.decoder_op,self.zeros)
        y_true = tf.where(self.sample,self.X,self.zeros)

        # self.cost1 = tf.multiply(self.X, tf.log(self.decoder_op))
        # self.cost2 = tf.multiply((1 - self.X), tf.log(1 - self.decoder_op))
        # self.loss = -1 * tf.multiply(self.X, tf.log(self.decoder_op)) - tf.multiply((1 - self.X), tf.log(1 - self.decoder_op))

        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred,labels=y_true)
        self.loss = tf.reduce_mean(self.loss)
        reg_lambda = tf.constant(self.regU, dtype=tf.float32)

        self.reg_loss = tf.add(tf.add(tf.multiply(reg_lambda, tf.nn.l2_loss(self.weights['encoder'])),
                               tf.multiply(reg_lambda, tf.nn.l2_loss(self.weights['decoder']))),
                               tf.add(tf.multiply(reg_lambda, tf.nn.l2_loss(self.biases['encoder'])),
                               tf.multiply(reg_lambda, tf.nn.l2_loss(self.biases['decoder']))))

        self.reg_loss = tf.add(self.reg_loss,tf.multiply(reg_lambda,tf.nn.l2_loss(self.V_embed)))
        self.loss = tf.add(self.loss,self.reg_loss)

        optimizer = tf.train.AdamOptimizer(self.lRate).minimize(self.loss)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        total_batch = int(len(self.dao.user)/ self.batch_size)
        for epoch in range(self.maxIter):
            for i in range(total_batch):
                batch_xs,users,sample = self.next_batch()

                _, loss = self.sess.run([optimizer, self.loss], feed_dict={self.X: batch_xs,self.v_idx:users,self.sample:sample})

                print self.foldInfo,"Epoch:", '%04d' % (epoch + 1),"Batch:", '%03d' %(i+1),"loss=", "{:.9f}".format(loss)
        print("Optimization Finished!")



    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            vec = self.dao.row(u).reshape((1,len(self.dao.item)))
            uid = [self.dao.user[u]]
            return self.sess.run(self.decoder_op,feed_dict={self.X:vec,self.v_idx:uid})[0]
        else:
            return [self.dao.globalMean] * len(self.dao.item)


