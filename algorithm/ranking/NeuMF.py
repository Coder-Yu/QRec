#coding:utf8
from baseclass.DeepRecommender import DeepRecommender
import numpy as np
from random import randint
import tensorflow as tf

class NeuMF(DeepRecommender):

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(NeuMF, self).__init__(conf,trainingSet,testSet,fold)


    def next_batch(self):
        batch_idx = np.random.randint(self.train_size, size=self.batch_size)
        users = [self.data.trainingData[idx][0] for idx in batch_idx]
        items = [self.data.trainingData[idx][1] for idx in batch_idx]
        user_idx,item_idx=[],[]
        y = []
        for i,user in enumerate(users):
            user_idx.append(self.data.user[user])
            item_idx.append(self.data.item[items[i]])
            y.append(1)
            #According to the paper, we sampled four negative instances per positive instance
            for instance in range(4):
                item_j = randint(0, self.num_items - 1)
                while self.data.trainSet_u[user].has_key(self.data.id2item[item_j]):
                    item_j = randint(0, self.num_items - 1)
                user_idx.append(self.data.user[user])
                item_idx.append(item_j)
                y.append(0)
        return user_idx,item_idx,y


    def initModel(self):
        super(NeuMF, self).initModel()
        # parameters used are consistent with default settings in the original paper
        mlp_regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("latent_factors"):
            self.PG = tf.get_variable(name='PG',initializer=initializer([self.num_users, self.embed_size]))
            self.QG = tf.get_variable(name='QG',initializer=initializer([self.num_items, self.embed_size]))

            self.PM = tf.get_variable(name='PM', initializer=initializer([self.num_users, self.embed_size]),regularizer=mlp_regularizer)
            self.QM = tf.get_variable(name='QM', initializer=initializer([self.num_items, self.embed_size]),regularizer=mlp_regularizer)

        with tf.name_scope("input"):
            self.r = tf.placeholder(tf.float32, [None], name="rating")
            self.u_idx = tf.placeholder(tf.int32, [None], name="u_idx")
            self.i_idx = tf.placeholder(tf.int32, [None], name="i_idx")
            self.UG_embedding = tf.nn.embedding_lookup(self.PG, self.u_idx)
            self.IG_embedding = tf.nn.embedding_lookup(self.QG, self.i_idx)
            self.UM_embedding = tf.nn.embedding_lookup(self.PM, self.u_idx)
            self.IM_embedding = tf.nn.embedding_lookup(self.QM, self.i_idx)

        # Generic Matrix Factorization
        with tf.variable_scope("mf_output"):
            self.GMF_Layer = tf.multiply(self.UG_embedding,self.IG_embedding)
            self.h_mf = tf.get_variable(name='mf_out', initializer=initializer([self.embed_size]))

        # MLP
        with tf.variable_scope("mlp_params"):
            MLP_W1 = tf.get_variable(name='W1',initializer=initializer([self.embed_size*2, self.embed_size*5]), regularizer=mlp_regularizer)
            MLP_b1 = tf.get_variable(name='b1',initializer=tf.zeros(shape=[self.embed_size*5]), regularizer=mlp_regularizer)
            self.h_out = tf.nn.relu(tf.add(tf.matmul(tf.concat([self.UM_embedding,self.IM_embedding], 1), MLP_W1), MLP_b1))

            MLP_W2 = tf.get_variable(name='W2',initializer=initializer([self.embed_size*5, self.embed_size*2]), regularizer=mlp_regularizer)
            MLP_b2 = tf.get_variable(name='b2',initializer=tf.zeros(shape=[self.embed_size*2]), regularizer=mlp_regularizer)
            self.h_out = tf.nn.relu(tf.add(tf.matmul(self.h_out,MLP_W2), MLP_b2))

            MLP_W3 = tf.get_variable(name='W3',initializer=initializer([self.embed_size*2, self.embed_size]),regularizer=mlp_regularizer)
            MLP_b3 = tf.get_variable(name='b3',initializer=tf.zeros(shape=[self.embed_size]), regularizer=mlp_regularizer)
            self.MLP_Layer = tf.nn.relu(tf.add(tf.matmul(self.h_out,MLP_W3), MLP_b3))
            self.h_mlp = tf.get_variable(name='mlp_out', initializer=initializer([self.embed_size]), regularizer=mlp_regularizer)


        #single inference
        #GMF
        self.y_mf = tf.reduce_sum(tf.multiply(self.GMF_Layer,self.h_mf),1)
        self.y_mf = tf.sigmoid(self.y_mf)
        self.mf_loss = self.r * tf.log(self.y_mf+10e-10) + (1 - self.r) * tf.log(1 - self.y_mf+10e-10)
        mf_reg = self.regU*(tf.nn.l2_loss(self.UG_embedding)+tf.nn.l2_loss(self.IG_embedding) + tf.nn.l2_loss(self.h_mf))

        self.mf_loss = -tf.reduce_sum(self.mf_loss) + mf_reg

        self.mf_optimizer = tf.train.AdamOptimizer(self.lRate).minimize(self.mf_loss)
        #MLP
        self.y_mlp = tf.reduce_sum(tf.multiply(self.MLP_Layer,self.h_mlp),1)
        self.y_mlp = tf.sigmoid(self.y_mlp)
        self.mlp_loss = self.r * tf.log(self.y_mlp+10e-10) + (1 - self.r) * tf.log(1 - self.y_mlp+10e-10)
        self.mlp_loss = -tf.reduce_sum(self.mlp_loss)
        self.mlp_optimizer = tf.train.AdamOptimizer(self.lRate).minimize(self.mlp_loss)

        #fusion
        self.NeuMF_Layer = tf.concat([self.GMF_Layer,self.MLP_Layer], 1)
        self.h_NeuMF = tf.concat([0.5*self.h_mf,0.5*self.h_mlp], 0)
        self.y_neu = tf.reduce_sum(tf.multiply(self.NeuMF_Layer, self.h_NeuMF), 1)
        self.y_neu = tf.sigmoid(self.y_neu)
        self.neu_loss = self.r * tf.log(self.y_neu+10e-10) + (1 - self.r) * tf.log(1 - self.y_neu+10e-10)

        self.neu_loss = -tf.reduce_sum(self.neu_loss)+ mf_reg + self.regU*tf.nn.l2_loss(self.h_NeuMF)
        ###it seems Adam is better than SGD here...
        self.neu_optimizer = tf.train.AdamOptimizer(self.lRate).minimize(self.neu_loss)

    def buildModel(self):

        init = tf.global_variables_initializer()
        self.sess.run(init)

        print 'pretraining... (GMF)'
        for epoch in range(self.maxIter):
            user_idx, item_idx, r = self.next_batch()

            _, loss,y_mf = self.sess.run([self.mf_optimizer, self.mf_loss,self.y_mf],
                               feed_dict={self.u_idx: user_idx, self.i_idx: item_idx, self.r: r})
            print 'iteration:', epoch, 'loss:', loss

        print 'pretraining... (MLP)'
        for epoch in range(self.maxIter/2):
            user_idx, item_idx, r = self.next_batch()
            _, loss, y_mlp = self.sess.run([self.mlp_optimizer, self.mlp_loss, self.y_mlp],
                                          feed_dict={self.u_idx: user_idx, self.i_idx: item_idx, self.r: r})
            print 'iteration:', epoch, 'loss:', loss

        print 'training... (NeuMF)'
        for epoch in range(self.maxIter/10):
            user_idx, item_idx, r = self.next_batch()
            _, loss, y_neu = self.sess.run([self.neu_optimizer, self.neu_loss, self.y_neu],
                                          feed_dict={self.u_idx: user_idx, self.i_idx: item_idx, self.r: r})
            print 'iteration:', epoch, 'loss:', loss

    def predict_mlp(self,uid):
        user_idx = [uid]*self.num_items
        y_mlp = self.sess.run([self.y_mlp],feed_dict={self.u_idx: user_idx, self.i_idx: range(self.num_items)})
        return y_mlp[0]

    def predict_mf(self,uid):
        user_idx = [uid]*self.num_items
        y_mf = self.sess.run([self.y_mf],feed_dict={self.u_idx: user_idx, self.i_idx: range(self.num_items)})
        return y_mf[0]

    def predict_neu(self,uid):
        user_idx = [uid]*self.num_items
        y_neu = self.sess.run([self.y_neu],feed_dict={self.u_idx: user_idx, self.i_idx: range(self.num_items)})
        return y_neu[0]

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.user[u]
            return self.predict_neu(u)
        else:
            return [self.data.globalMean] * self.num_items