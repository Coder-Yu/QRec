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

#According to the paper, we only
class DMF(IterativeRecommender):

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(DMF, self).__init__(conf,trainingSet,testSet,fold)


    def next_batch(self,i):
        rows = np.zeros(((self.negative_sp+1)*self.batch_size,self.n))
        cols = np.zeros(((self.negative_sp+1)*self.batch_size,self.m))
        batch_idx = range(self.batch_size*i,self.batch_size*(i+1))

        users = [self.dao.trainingData[idx][0] for idx in batch_idx]
        items = [self.dao.trainingData[idx][1] for idx in batch_idx]
        u_idx = [self.dao.user[u] for u in users]
        v_idx = [self.dao.item[i] for i in items]
        ratings = [float(self.dao.trainingData[idx][2]) for idx in batch_idx]

        for i,user in enumerate(users):
            rows[i] = self.dao.row(user)
        for i,item in enumerate(items):
            cols[i] = self.dao.col(item)

        #negative sample
        for i in range(self.negative_sp*self.batch_size):
            u = choice(self.dao.user)
            v = choice(self.dao.item)
            while self.dao.contains(u,v):
                u = choice(self.dao.user)
                v = choice(self.dao.item)
            rows[self.batch_size-1+i]=self.dao.row(u)
            cols[self.batch_size-1+i]=self.dao.col(i)
            u_idx.append(self.dao.user[u])
            v_idx.append(self.dao.item[u])
            ratings.append(0)
        return rows,cols,ratings,u_idx,v_idx

    def initModel(self):
        super(DMF, self).initModel()
        n_input_u = len(self.dao.item)
        n_input_i = len(self.dao.user)
        self.negative_sp = 5
        self.n_hidden_u=[64,128]
        self.n_hidden_i=[64,128]
        self.input_u = tf.placeholder("float", [None, n_input_u])
        self.input_i = tf.placeholder("float", [None, n_input_i])


    def buildModel_tf(self):
        super(DMF, self).buildModel_tf()

        initializer = tf.contrib.layers.xavier_initializer()
        #user net
        user_W1 = tf.Variable(initializer([self.n, self.n_hidden_u[0]]))
        self.user_out = tf.matmul(self.input_u, user_W1)
        self.regLoss = tf.nn.l2_loss(user_W1)
        for i in range(1, len(self.n_hidden_u)):
            W = tf.Variable(initializer([self.n_hidden_u[i-1], self.n_hidden_u[i]]))
            b = tf.Variable(initializer([self.n_hidden_u[i]]))
            self.regLoss = tf.add(self.regLoss,tf.nn.l2_loss(W))
            self.regLoss = tf.add(self.regLoss, tf.nn.l2_loss(b))
            self.user_out = tf.nn.relu(tf.add(tf.matmul(self.user_out, W), b))

        #item net
        item_W1 = tf.Variable(initializer([self.m, self.n_hidden_i[0]]))
        self.item_out = tf.matmul(self.input_i, item_W1)
        self.regLoss = tf.add(self.regLoss, tf.nn.l2_loss(item_W1))
        for i in range(1, len(self.n_hidden_i)):
            W = tf.Variable(initializer([self.n_hidden_i[i-1], self.n_hidden_i[i]]))
            b = tf.Variable(initializer([self.n_hidden_i[i]]))
            self.regLoss = tf.add(self.regLoss, tf.nn.l2_loss(W))
            self.regLoss = tf.add(self.regLoss, tf.nn.l2_loss(W))
            self.item_out = tf.nn.relu(tf.add(tf.matmul(self.item_out, W), b))

        norm_user_output = tf.sqrt(tf.reduce_sum(tf.square(self.user_out), axis=1))
        norm_item_output = tf.sqrt(tf.reduce_sum(tf.square(self.item_out), axis=1))

        self.y_ = tf.reduce_sum(tf.multiply(self.user_out, self.item_out), axis=1) / (
                norm_item_output * norm_user_output)
        self.y_ = tf.maximum(1e-6, self.y_)

        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_,labels=self.r)
        self.loss = tf.reduce_mean(self.loss)
        reg_lambda = tf.constant(self.regU, dtype=tf.float32)
        self.regLoss = tf.multiply(reg_lambda,self.regLoss)
        self.loss = tf.add(self.loss,self.regLoss)

        optimizer = tf.train.AdamOptimizer(self.lRate).minimize(self.loss)

        self.U = np.zeros((self.m, self.n_hidden_u[-1]))
        self.V = np.zeros((self.n, self.n_hidden_u[-1]))

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        total_batch = int(len(self.dao.trainingData)/ self.batch_size)
        for epoch in range(self.maxIter):
            for i in range(total_batch):
                users,items,ratings,u_idx,v_idx = self.next_batch(i)

                _,loss= self.sess.run([optimizer, self.loss], feed_dict={self.input_u: users,self.input_i:items,self.r:ratings})

                #save the output layer
                if epoch == self.maxIter-1:
                    U_embedding, V_embedding = self.sess.run([self.user_out, self.item_out], feed_dict={self.input_u: users,self.input_i:items})
                    for ue,u in zip(U_embedding,u_idx):
                        self.U[u]=ue
                    for ve,v in zip(V_embedding,v_idx):
                        self.V[v]=ve

                print self.foldInfo,"Epoch:", '%04d' % (epoch + 1),"Batch:", '%03d' %(i+1),"loss=", "{:.9f}".format(loss)

        print("Optimization Finished!")

        self.normalized_V = np.sqrt(np.sum(self.V*self.V,axis=1))
        self.normalized_U = np.sqrt(np.sum(self.U*self.U,axis=1))


    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            uid = self.dao.user[u]
            return np.divide(self.V.dot(self.U[uid]),self.normalized_U[uid]*self.normalized_V)
        else:
            return [self.dao.globalMean] * self.n


