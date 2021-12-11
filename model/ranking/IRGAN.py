#coding:utf8
from base.deepRecommender import DeepRecommender
import numpy as np
from random import randint,choice
import tensorflow as tf
###
# We just transformed the code released by the authors to RecQ and slightly modified the code in order to invoke
# the interfaces of RecQ
###
class GEN():
    def __init__(self, itemNum, userNum, emb_dim, lamda, learning_rate=0.002):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.g_params = []
        with tf.variable_scope('generator'):
            self.user_embeddings = tf.Variable(
                tf.random_uniform([self.userNum, self.emb_dim], minval=-0.05, maxval=0.05, dtype=tf.float32))
            self.item_embeddings = tf.Variable(
                tf.random_uniform([self.itemNum, self.emb_dim], minval=-0.05, maxval=0.05, dtype=tf.float32))
            self.item_bias = tf.Variable(tf.zeros([self.itemNum]))
            self.g_params = [self.user_embeddings, self.item_embeddings, self.item_bias]

        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)
        self.label = tf.placeholder(tf.float32)
        self.reward = tf.placeholder(tf.float32)
        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i)
        self.i_bias = tf.gather(self.item_bias, self.i)
        self.pre_train_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label,
                            logits=tf.reduce_sum(tf.multiply(self.u_embedding,self.i_embedding),1))
        self.pre_train_loss += self.lamda * (tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding)
                                             + tf.nn.l2_loss(self.i_bias))
        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias
        self.i_prob = tf.gather(tf.reshape(tf.nn.softmax(tf.reshape(self.all_logits, [1, -1])), [-1]),self.i)
        self.gan_loss = -tf.reduce_mean(tf.log(self.i_prob) * self.reward) + self.lamda * (
            tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding) + tf.nn.l2_loss(self.i_bias))
        g_opt = tf.train.AdamOptimizer(learning_rate)
        self.gan_updates = g_opt.minimize(self.gan_loss, var_list=self.g_params)

class DIS():
    def __init__(self, itemNum, userNum, emb_dim, lamda, learning_rate=0.002):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.d_params = []
        with tf.variable_scope('discriminator'):
            self.user_embeddings = tf.Variable(
                tf.random_uniform([self.userNum, self.emb_dim], minval=-0.05, maxval=0.05,dtype=tf.float32))
            self.item_embeddings = tf.Variable(
                tf.random_uniform([self.itemNum, self.emb_dim], minval=-0.05, maxval=0.05,dtype=tf.float32))
            self.item_bias = tf.Variable(tf.zeros([self.itemNum]))
        self.d_params = [self.user_embeddings, self.item_embeddings, self.item_bias]

        # placeholder definition
        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)
        self.label = tf.placeholder(tf.float32)
        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i)
        self.i_bias = tf.gather(self.item_bias, self.i)
        self.pre_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), 1) + self.i_bias
        self.pre_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.pre_logits) \
                        + self.lamda * (tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding)
                                        + tf.nn.l2_loss(self.i_bias))

        d_opt = tf.train.AdamOptimizer(learning_rate)
        self.d_updates = d_opt.minimize(self.pre_loss, var_list=self.d_params)
        self.reward_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding),1) + self.i_bias
        self.reward = 2 * (tf.sigmoid(self.reward_logits) - 0.5)
        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias
        self.NLL = -tf.reduce_mean(tf.log(tf.gather(tf.reshape(tf.nn.softmax(tf.reshape(self.all_logits, [1, -1])), [-1]), self.i)))

class IRGAN(DeepRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(IRGAN, self).__init__(conf,trainingSet,testSet,fold)

    def get_data(self,model):
        users,items,label = [],[],[]
        for user in self.data.trainSet_u:
            pos,values = self.data.userRated(user)
            pos = [self.data.item[item] for item in pos]
            u = self.data.user[user]
            rating = self.sess.run(model.all_logits, {model.u: [u]})
            rating = np.array(rating) / 0.2  # Temperature
            exp_rating = np.exp(rating)
            exp_rating[np.array(pos)] = 0
            prob = exp_rating / np.sum(exp_rating)
            neg = np.random.choice(np.arange(self.num_items), size=2*len(pos), p=prob)
            for i in range(len(pos)):
                users.append(u)
                items.append(pos[i])
                label.append(1.)
            for i in range(len(neg)):
                users.append(u)
                items.append(neg[i])
                label.append(0.)
        return users,items,label

    def get_batch(self,data,index,size):
        user,item,label = data
        return (user[index:index+size],item[index:index+size],label[index:index+size])

    def initModel(self):
        super(IRGAN, self).initModel()
        self.generator = GEN(self.num_items, self.num_users, self.emb_size, lamda=self.regU, learning_rate=self.lRate)
        self.discriminator = DIS(self.num_items, self.num_users, self.emb_size, lamda=self.regU, learning_rate=self.lRate)

    def trainModel(self):
        # minimax training
        init = tf.global_variables_initializer()
        self.sess.run(init)
        #pretrain the discriminator
        # for i in range(100):
        #     input_user, input_item, input_label = self.next_batch()
        #     _ = self.sess.run(self.discriminator.d_updates,
        #                       feed_dict={self.discriminator.u: input_user, self.discriminator.i: input_item,
        #                                  self.discriminator.label: input_label})
        for epoch in range(self.maxEpoch):
            print('Update discriminator...')
            for d_epoch in range(1):
                user_set,item_set,labels = self.get_data(self.generator)
                data = [user_set,item_set,labels]
                index = 0
                while index<self.train_size:
                    if index + self.batch_size <= self.train_size:
                        input_user, input_item, input_label = self.get_batch(data, index, self.batch_size)
                    else:
                        input_user, input_item, input_label = self.get_batch(data, index, self.train_size - index)
                    index += self.batch_size
                    _ = self.sess.run(self.discriminator.d_updates,
                                 feed_dict={self.discriminator.u: input_user, self.discriminator.i: input_item,
                                            self.discriminator.label: input_label})
                print('epoch:',epoch+1,'d_epoch:', d_epoch+1)

            # Train G
            print('Update generator...')
            for g_epoch in range(5):
                for user in self.data.trainSet_u:
                    sample_lambda = 0.2
                    pos, values = self.data.userRated(user)
                    pos = [self.data.item[item] for item in pos]
                    u = self.data.user[user]
                    rating = self.sess.run(self.generator.all_logits, {self.generator.u: u})
                    exp_rating = np.exp(rating)
                    prob = exp_rating / np.sum(exp_rating)  # prob is generator distribution p_\theta
                    # Here is the importance sampling. Actually I have some problems in understandings these two
                    # lines and the paper doesn't give details about the importance sampling.
                    pn = (1 - sample_lambda) * prob
                    pn[pos] += sample_lambda * 1.0 / len(pos)
                    # Now, pn is the Pn in importance sampling, prob is generator distribution p_\theta

                    sample = np.random.choice(np.arange(self.num_items), 3 * len(pos), p=pn)
                    ###########################################################################
                    # Get reward and adapt it with importance sampling
                    ###########################################################################
                    reward = self.sess.run(self.discriminator.reward,
                                           {self.discriminator.u: u, self.discriminator.i: sample})
                    reward = reward * prob[sample] / pn[sample]
                    ###########################################################################
                    # Update G
                    ###########################################################################
                    _ = self.sess.run(self.generator.gan_updates,
                                      {self.generator.u: u, self.generator.i: sample,
                                       self.generator.reward: reward})

                print('epoch:', epoch+1, 'g_epoch:', g_epoch+1)

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.user[u]

            #In our experiments, discriminator performs better than generator
            res = self.sess.run(self.discriminator.all_logits, {self.discriminator.u: [u]})
            return res

        else:
            return [self.data.globalMean] * self.num_items



