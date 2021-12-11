#coding:utf8
from base.deepRecommender import DeepRecommender
import numpy as np
from random import randint,choice
import tensorflow as tf



class CFGAN(DeepRecommender):

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(CFGAN, self).__init__(conf,trainingSet,testSet,fold)
        #It is quite interesting and confusing that when I set ratio_zr = 0 and ratio_zp = 0, CFGAN reaches the best performance.
        self.S_zr = 0.001
        self.S_pm = 0.001
        self.alpha = 0.01

    def next_batch(self):
        C_u = np.zeros((self.batch_size,self.num_items))
        N_zr = np.zeros((self.batch_size,self.num_items))
        mask = np.zeros((self.batch_size,self.num_items)) #e_u + k_u
        userList = list(self.data.user.keys())
        itemList = list(self.data.item.keys())
        for n in range(self.batch_size):
            user = choice(userList)
            vec = self.data.row(user)
            ratedItems, values = self.data.userRated(user)
            for item in ratedItems:
                iid = self.data.item[item]
                mask[n][iid]=1
            for i in range(int(self.S_zr*self.num_items)):
                ng = choice(itemList)
                while ng in self.data.trainSet_u[user]:
                    ng = choice(itemList)
                ng = self.data.item[ng]
                N_zr[n][ng] = 1
            for i in range(int(self.S_pm*self.num_items)):
                ng = choice(itemList)
                while ng in self.data.trainSet_u[user]:
                    ng = choice(itemList)
                ng = self.data.item[ng]
                mask[n][ng] = 1
            C_u[n]=vec
        return C_u,mask,N_zr

    def initModel(self):
        super(CFGAN, self).initModel()
        G_regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
        D_regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
        xavier_init = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("Generator"):
            # Generator Net
            self.C = tf.placeholder(tf.float32, shape=[None, self.num_items], name='C')
            G_W1 = tf.get_variable(name='G_W1',initializer=xavier_init([self.num_items,self.num_items]), regularizer=G_regularizer)
            G_b1 = tf.get_variable(name='G_b1',initializer=tf.zeros(shape=[self.num_items]), regularizer=G_regularizer)

            # G_W2 = tf.get_variable(name='G_W2',initializer=xavier_init([300,200]), regularizer=G_regularizer)
            # G_b2 = tf.get_variable(name='G_b2',initializer=tf.zeros(shape=[200]), regularizer=G_regularizer)
            #
            # G_W3 = tf.get_variable(initializer=xavier_init([200,self.num_items]), name='G_W3',regularizer=G_regularizer)
            # G_b3 = tf.get_variable(initializer=tf.zeros(shape=[self.num_items]), name='G_b3',regularizer=G_regularizer)

            theta_G = [G_W1, G_b1]#G_W2, G_W3, G_b1, G_b2, G_b3]

        with tf.variable_scope("Discriminator"):
            # Discriminator Net
            self.X = tf.placeholder(tf.float32, shape=[None, self.num_items], name='X')
            D_W1 = tf.get_variable(initializer=xavier_init([self.num_items*2,1]), name='D_W1',regularizer=D_regularizer)
            D_b1 = tf.get_variable(initializer=tf.zeros(shape=[1]), name='D_b1',regularizer=D_regularizer)

            # D_W2 = tf.get_variable(name='D_W2', initializer=xavier_init([300, 200]), regularizer=D_regularizer)
            # D_b2 = tf.get_variable(name='D_b2', initializer=tf.zeros(shape=[200]), regularizer=D_regularizer)
            #
            # D_W3 = tf.get_variable(initializer=xavier_init([200,1]), name='D_W3',regularizer=D_regularizer)
            # D_b3 = tf.get_variable(initializer=tf.zeros(shape=[1]), name='D_b3',regularizer=D_regularizer)

            theta_D = [D_W1, D_b1]#D_W2, D_W3, D_b1, D_b2, D_b3]

        self.mask = tf.placeholder(tf.float32, shape=[None, self.num_items], name='mask')
        self.N_zr = tf.placeholder(tf.float32, shape=[None, self.num_items], name='mask')

        #inference
        def generator():
            r_hat = tf.nn.sigmoid(tf.matmul(self.C, G_W1) + G_b1)
            # G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
            # r_hat = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
            fake_data = tf.multiply(r_hat,self.mask)
            return fake_data

        def discriminator(x):
            D_output = tf.nn.sigmoid(tf.matmul(x, D_W1) + D_b1)
            # D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
            # D_output = tf.nn.sigmoid(tf.matmul(D_h2, D_W3) + D_b3)
            return  D_output

        def r_hat():
            r_hat = tf.nn.sigmoid(tf.matmul(self.C, G_W1) + G_b1)
            # G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
            # r_hat = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
            return r_hat
        G_sample = generator()
        self.r_hat = r_hat()
        D_real = discriminator(tf.concat([self.C,self.C],1))
        D_fake = discriminator(tf.concat([G_sample,self.C],1))
        self.D_loss = -tf.reduce_mean(tf.log(D_real+10e-5) + tf.log(1. - D_fake+10e-5))
        self.G_loss = tf.reduce_mean(tf.log(1.-D_fake+10e-5)+self.alpha*tf.nn.l2_loss(tf.multiply(self.N_zr,G_sample)))

        # Only update D(X)'s parameters, so var_list = theta_D
        self.D_solver = tf.train.AdamOptimizer(self.lRate).minimize(self.D_loss, var_list=theta_D)
        # Only update G(X)'s parameters, so var_list = theta_G
        self.G_solver = tf.train.AdamOptimizer(self.lRate).minimize(self.G_loss, var_list=theta_G)


    def trainModel(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print('pretraining...')
        print('training...')
        for epoch in range(self.maxEpoch):
            G_loss = 0
            C_u, mask, N_zr = self.next_batch()
            _, D_loss = self.sess.run([self.D_solver, self.D_loss], feed_dict={self.C: C_u,self.mask:mask,self.N_zr:N_zr})
            for i in range(3):
                _, G_loss = self.sess.run([self.G_solver, self.G_loss], feed_dict={self.C: C_u,self.mask:mask,self.N_zr:N_zr})
            #C_u, mask, N_u = self.next_batch()
            print('epoch:', epoch, 'D_loss:', D_loss, 'G_loss', G_loss)

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            vec = self.data.row(u).reshape(1,self.num_items)
            res = self.sess.run([self.r_hat], feed_dict={self.C: vec})[0]
            return res[0]
        else:
            return [self.data.globalMean] * self.num_items



