#coding:utf8
from base.deepRecommender import DeepRecommender
import numpy as np
from random import choice,random,randint,shuffle
from util import config
import tensorflow as tf


#According to the paper, we only
class DMF(DeepRecommender):

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(DMF, self).__init__(conf,trainingSet,testSet,fold)

    def next_batch(self,i):
        rows = np.zeros(((self.negative_sp+1)*self.batch_size,self.num_items))
        cols = np.zeros(((self.negative_sp+1)*self.batch_size,self.num_users))
        batch_idx = list(range(self.batch_size*i,self.batch_size*(i+1)))
        users = [self.data.trainingData[idx][0] for idx in batch_idx]
        items = [self.data.trainingData[idx][1] for idx in batch_idx]
        u_idx = [self.data.user[u] for u in users]
        v_idx = [self.data.item[i] for i in items]
        ratings = [float(self.data.trainingData[idx][2]) for idx in batch_idx]
        for i,user in enumerate(users):
            rows[i] = self.data.row(user)
        for i,item in enumerate(items):
            cols[i] = self.data.col(item)
        userList = list(self.data.user.keys())
        itemList = list(self.data.item.keys())
        #negative sample
        for i in range(self.negative_sp*self.batch_size):
            u = choice(userList)
            v = choice(itemList)
            while self.data.contains(u,v):
                u = choice(userList)
                v = choice(itemList)
            rows[self.batch_size-1+i]=self.data.row(u)
            cols[self.batch_size-1+i]=self.data.col(i)
            u_idx.append(self.data.user[u])
            v_idx.append(self.data.item[v])
            ratings.append(0)
        return rows,cols,np.array(ratings),np.array(u_idx),np.array(v_idx)

    def initModel(self):
        super(DMF, self).initModel()
        n_input_u = len(self.data.item)
        n_input_i = len(self.data.user)
        self.negative_sp = 5
        self.n_hidden_u=[256,512]
        self.n_hidden_i=[256,512]
        self.input_u = tf.placeholder(tf.float, [None, n_input_u])
        self.input_i = tf.placeholder(tf.float, [None, n_input_i])

    def trainModel(self):
        super(DMF, self).trainModel_tf()
        initializer = tf.contrib.layers.xavier_initializer()
        #user net
        user_W1 = tf.Variable(initializer([self.num_items, self.n_hidden_u[0]],stddev=0.01))
        self.user_out = tf.nn.relu(tf.matmul(self.input_u, user_W1))
        self.regLoss = tf.nn.l2_loss(user_W1)
        for i in range(1, len(self.n_hidden_u)):
            W = tf.Variable(initializer([self.n_hidden_u[i-1], self.n_hidden_u[i]],stddev=0.01))
            b = tf.Variable(initializer([self.n_hidden_u[i]],stddev=0.01))
            self.regLoss = tf.add(self.regLoss,tf.nn.l2_loss(W))
            self.regLoss = tf.add(self.regLoss, tf.nn.l2_loss(b))
            self.user_out = tf.nn.relu(tf.add(tf.matmul(self.user_out, W), b))
        #item net
        item_W1 = tf.Variable(initializer([self.num_users, self.n_hidden_i[0]],stddev=0.01))
        self.item_out = tf.nn.relu(tf.matmul(self.input_i, item_W1))
        self.regLoss = tf.add(self.regLoss, tf.nn.l2_loss(item_W1))
        for i in range(1, len(self.n_hidden_i)):
            W = tf.Variable(initializer([self.n_hidden_i[i-1], self.n_hidden_i[i]],stddev=0.01))
            b = tf.Variable(initializer([self.n_hidden_i[i]],stddev=0.01))
            self.regLoss = tf.add(self.regLoss, tf.nn.l2_loss(W))
            self.regLoss = tf.add(self.regLoss, tf.nn.l2_loss(b))
            self.item_out = tf.nn.relu(tf.add(tf.matmul(self.item_out, W), b))
        norm_user_output = tf.sqrt(tf.reduce_sum(tf.square(self.user_out), axis=1))
        norm_item_output = tf.sqrt(tf.reduce_sum(tf.square(self.item_out), axis=1))
        self.y_ = tf.reduce_sum(tf.multiply(self.user_out, self.item_out), axis=1) / (
                norm_item_output * norm_user_output)
        self.y_ = tf.maximum(1e-6, self.y_)
        self.loss = self.r*tf.log(self.y_) + (1 - self.r) * tf.log(1 - self.y_)#tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_,labels=self.r)
        #self.loss = tf.nn.l2_loss(tf.subtract(self.y_,self.r))
        self.loss = -tf.reduce_sum(self.loss)
        reg_lambda = tf.constant(self.regU, dtype=tf.float32)
        self.regLoss = tf.multiply(reg_lambda,self.regLoss)
        self.loss = tf.add(self.loss,self.regLoss)
        optimizer = tf.train.AdamOptimizer(self.lRate).minimize(self.loss)
        self.U = np.zeros((self.num_users, self.n_hidden_u[-1]))
        self.V = np.zeros((self.num_items, self.n_hidden_u[-1]))
        init = tf.global_variables_initializer()
        self.sess.run(init)

        total_batch = int(len(self.data.trainingData)/ self.batch_size)
        for epoch in range(self.maxEpoch):
            shuffle(self.data.trainingData)
            for i in range(total_batch):
                users,items,ratings,u_idx,v_idx = self.next_batch(i)
                shuffle_idx=np.random.permutation(list(range(len(users))))
                users = users[shuffle_idx]
                items = items[shuffle_idx]
                ratings = ratings[shuffle_idx]
                u_idx = u_idx[shuffle_idx]
                v_idx = v_idx[shuffle_idx]
                _,loss= self.sess.run([optimizer, self.loss], feed_dict={self.input_u: users,self.input_i:items,self.r:ratings})
                print(self.foldInfo, "Epoch:", '%04d' % (epoch + 1), "Batch:", '%03d' % (i + 1), "loss=", "{:.9f}".format(loss))
            #save the output layer
                U_embedding, V_embedding = self.sess.run([self.user_out, self.item_out], feed_dict={self.input_u: users,self.input_i:items})
                for ue,u in zip(U_embedding,u_idx):
                    self.U[u]=ue
                for ve,v in zip(V_embedding,v_idx):
                    self.V[v]=ve
            self.normalized_V = np.sqrt(np.sum(self.V * self.V, axis=1))
            self.normalized_U = np.sqrt(np.sum(self.U * self.U, axis=1))
        print("Optimization Finished!")

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            uid = self.data.user[u]
            return np.divide(self.V.dot(self.U[uid]),self.normalized_U[uid]*self.normalized_V)
        else:
            return [self.data.globalMean] * self.num_items


