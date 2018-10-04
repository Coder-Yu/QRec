#coding:utf8
from baseclass.DeepRecommender import DeepRecommender
import numpy as np
from random import choice,random
from tool import config
try:
    import tensorflow as tf
except ImportError:
    print 'This method can only be run tensorflow!'
    exit(-1)
from tensorflow import set_random_seed
set_random_seed(2)

class CDAE(DeepRecommender):

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(CDAE, self).__init__(conf,trainingSet,testSet,fold)

    def encoder(self,x,v):
        layer = tf.nn.sigmoid(tf.add(tf.add(tf.matmul(x, self.weights['encoder']), self.biases['encoder']),v))
        #layer = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder']), self.biases['encoder']))
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

            ratedItems, values = self.dao.userRated(user)
            for item in ratedItems:
                iid = self.dao.item[item]
                evaluated[n][iid]=True
            for i in range(self.negative_sp*len(ratedItems)):
                ng = choice(itemList)
                while self.dao.trainSet_u.has_key(ng):
                    ng = choice(itemList)
                ng = self.dao.item[ng]
                evaluated[n][ng]=True
            X[n]=vec
        return X,uids,evaluated

    def readConfiguration(self):
        super(CDAE, self).readConfiguration()
        eps = config.LineConfig(self.config['CDAE'])
        self.corruption_level = float(eps['-co'])
        self.n_hidden = int(eps['-nh'])

    def initModel(self):
        super(CDAE, self).initModel()
        n_input = len(self.dao.item)
        n_output = len(self.dao.item)
        self.negative_sp = 5
        initializer = tf.contrib.layers.xavier_initializer()
        self.X = tf.placeholder("float", [None, n_input])
        self.mask_corruption = tf.placeholder("float", [None, n_input])
        self.sample = tf.placeholder("bool", [None, n_input])
        self.zeros = np.zeros((self.batch_size,n_input))
        self.V = tf.Variable(initializer([len(self.dao.user), self.n_hidden]))
        self.v_idx = tf.placeholder(tf.int32, [None], name="v_idx")
        self.V_embed = tf.nn.embedding_lookup(self.V, self.v_idx)


        self.weights = {
            'encoder': tf.Variable(initializer([n_input, self.n_hidden])),
            'decoder': tf.Variable(initializer([self.n_hidden, n_output])),
        }
        self.biases = {
            'encoder': tf.Variable(initializer([self.n_hidden])),
            'decoder': tf.Variable(initializer([n_output])),
        }

    #def pretrain(self,var,data):

    def buildModel_tf(self):
        self.corrupted_input = tf.multiply(self.X,self.mask_corruption)
        self.encoder_op = self.encoder(self.corrupted_input,self.V_embed)
        self.decoder_op = self.decoder(self.encoder_op)


        y_pred = tf.where(self.sample,self.decoder_op,self.zeros)
        y_true = tf.where(self.sample,self.corrupted_input,self.zeros)

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
                mask = np.random.binomial(1, self.corruption_level,(self.batch_size, len(self.dao.item)))
                batch_xs,users,sample = self.next_batch()

                _, loss = self.sess.run([optimizer, self.loss], feed_dict={self.X: batch_xs,self.mask_corruption:mask,self.v_idx:users,self.sample:sample})

                print self.foldInfo,"Epoch:", '%04d' % (epoch + 1),"Batch:", '%03d' %(i+1),"loss=", "{:.9f}".format(loss)
            self.ranking_performance()
        print("Optimization Finished!")



    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            vec = self.dao.row(u).reshape((1,len(self.dao.item)))
            uid = [self.dao.user[u]]
            return self.sess.run(self.decoder_op,feed_dict={self.X:vec,self.mask_corruption:np.ones((1,len(self.dao.item))),self.v_idx:uid})[0]
        else:
            return [self.dao.globalMean] * len(self.dao.item)


