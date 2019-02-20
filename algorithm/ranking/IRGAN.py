#coding:utf8
from baseclass.DeepRecommender import DeepRecommender
import numpy as np
from random import randint,choice
try:
    import tensorflow as tf
except ImportError:
    print 'This method can only run on tensorflow!'
    exit(-1)
from tensorflow import set_random_seed
set_random_seed(2)

###

# We just transformed the code released by the authors to RecQ and slightly modified the code in order to invoke
# the interfaces of RecQ

###
class GEN():
    def __init__(self, itemNum, userNum, emb_dim, lamda, param=None, initdelta=0.05, learning_rate=0.05):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.param = param
        self.initdelta = initdelta
        self.learning_rate = learning_rate
        self.g_params = []

        with tf.variable_scope('generator'):
            if self.param == None:
                self.user_embeddings = tf.Variable(
                    tf.random_uniform([self.userNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_embeddings = tf.Variable(
                    tf.random_uniform([self.itemNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_bias = tf.Variable(tf.zeros([self.itemNum]))
            else:
                self.user_embeddings = tf.Variable(self.param[0])
                self.item_embeddings = tf.Variable(self.param[1])
                self.item_bias = tf.Variable(param[2])

            self.g_params = [self.user_embeddings, self.item_embeddings, self.item_bias]

        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)
        self.reward = tf.placeholder(tf.float32)

        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i)
        self.i_bias = tf.gather(self.item_bias, self.i)

        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias
        self.i_prob = tf.gather(
            tf.reshape(tf.nn.softmax(tf.reshape(self.all_logits, [1, -1])), [-1]),
            self.i)

        self.gan_loss = -tf.reduce_mean(tf.log(self.i_prob) * self.reward) + self.lamda * (
            tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding) + tf.nn.l2_loss(self.i_bias))

        g_opt = tf.train.AdamOptimizer(self.learning_rate)
        self.gan_updates = g_opt.minimize(self.gan_loss, var_list=self.g_params)

        # for test stage, self.u: [self.batch_size]
        self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
                                    transpose_b=True) + self.item_bias

    # def save_model(self, sess, filename):
    #     param = sess.run(self.g_params)
    #     cPickle.dump(param, open(filename, 'w'))
    

class DIS():
    def __init__(self, itemNum, userNum, emb_dim, lamda, param=None, initdelta=0.05, learning_rate=0.05):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.param = param
        self.initdelta = initdelta
        self.learning_rate = learning_rate
        self.d_params = []

        with tf.variable_scope('discriminator'):
            if self.param == None:
                self.user_embeddings = tf.Variable(
                    tf.random_uniform([self.userNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_embeddings = tf.Variable(
                    tf.random_uniform([self.itemNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_bias = tf.Variable(tf.zeros([self.itemNum]))
            else:
                self.user_embeddings = tf.Variable(self.param[0])
                self.item_embeddings = tf.Variable(self.param[1])
                self.item_bias = tf.Variable(self.param[2])

        self.d_params = [self.user_embeddings, self.item_embeddings, self.item_bias]

        # placeholder definition
        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)
        self.label = tf.placeholder(tf.float32)

        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i)
        self.i_bias = tf.gather(self.item_bias, self.i)

        self.pre_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), 1) + self.i_bias
        self.pre_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label,
                                                                logits=self.pre_logits) + self.lamda * (
            tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding) + tf.nn.l2_loss(self.i_bias)
        )

        d_opt = tf.train.AdamOptimizer(self.learning_rate)
        self.d_updates = d_opt.minimize(self.pre_loss, var_list=self.d_params)

        self.reward_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding),
                                           1) + self.i_bias
        self.reward = 2 * (tf.sigmoid(self.reward_logits) - 0.5)

        # for test stage, self.u: [self.batch_size]
        self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
                                    transpose_b=True) + self.item_bias

        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias
        self.NLL = -tf.reduce_mean(tf.log(
            tf.gather(tf.reshape(tf.nn.softmax(tf.reshape(self.all_logits, [1, -1])), [-1]), self.i))
        )
        # for dns sample
        self.dns_rating = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias

    # def save_model(self, sess, filename):
    #     param = sess.run(self.d_params)
    #     cPickle.dump(param, open(filename, 'w'))




class IRGAN(DeepRecommender):

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(IRGAN, self).__init__(conf,trainingSet,testSet,fold)


    def get_data(self,model):

        user_list,item,label = [],[],[]
        for user in self.data.trainSet_u:
            pos,values = self.data.userRated(user)
            u = self.data.user[user]
            rating = self.sess.run(model.all_rating, {model.u: [u]})
            rating = np.array(rating[0]) / 0.2  # Temperature
            exp_rating = np.exp(rating)
            prob = exp_rating / np.sum(exp_rating)

            neg = np.random.choice(np.arange(self.n), size=len(pos), p=prob)
            for i in range(len(pos)):
                user_list.append(u)
                item.append(self.data.item[pos[i]])
                label.append(1)
            for i in range(len(neg)):
                user_list.append(u)
                item.append(neg[i])
                label.append(0)

        return (user_list,item,label),len(user_list)

    def get_batch(self,data,index,size):
        user,item,neg_item = data
        return (user[index:index+size],item[index:index+size],neg_item[index:index+size])


    def initModel(self):
        super(IRGAN, self).initModel()
        self.generator = GEN(self.n, self.m, self.k, lamda=self.regU, param=None,
                             initdelta=0.05,learning_rate=self.lRate)
        self.discriminator = DIS(self.n, self.m, self.k, lamda=self.regU, param=None,
                                 initdelta=0.05,learning_rate=self.lRate)



    def buildModel(self):
        # minimax training
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.maxIter):
            print 'Update discriminator...'
            for d_epoch in range(20):
                if d_epoch % 5 == 0:
                    data,train_size = self.get_data(self.generator)
                index = 0
                while True:
                    if index > train_size:
                        break
                    if index + self.batch_size <= train_size:
                        input_user, input_item, input_label = self.get_batch(data, index, self.batch_size)
                    else:
                        input_user, input_item, input_label = self.get_batch(data, index, train_size - index)

                    index += self.batch_size

                    _ = self.sess.run(self.discriminator.d_updates,
                                 feed_dict={self.discriminator.u: input_user, self.discriminator.i: input_item,
                                            self.discriminator.label: input_label})

                print 'epoch:',epoch,'d_epoch:', d_epoch


            # Train G
            print 'Update generator...'
            for g_epoch in range(10):
                for user in self.data.trainSet_u:
                    sample_lambda = 0.2
                    pos,values = self.data.userRated(user)
                    pos = [self.data.item[item] for item in pos]
                    u = self.data.user[user]
                    rating = self.sess.run(self.generator.all_logits, {self.generator.u: u})

                    exp_rating = np.exp(rating)
                    prob = exp_rating / np.sum(exp_rating)  # prob is generator distribution p_\theta

                    pn = (1 - sample_lambda) * prob
                    pn[pos] += sample_lambda * 1.0 / len(pos)
                    # Now, pn is the Pn in importance sampling, prob is generator distribution p_\theta

                    sample = np.random.choice(np.arange(self.n), 2 * len(pos), p=pn)
                    ###########################################################################
                    # Get reward and adapt it with importance sampling
                    ###########################################################################
                    reward = self.sess.run(self.discriminator.reward, {self.discriminator.u: u, self.discriminator.i: sample})
                    reward = reward * prob[sample] / pn[sample]
                    ###########################################################################
                    # Update G
                    ###########################################################################
                    _ = self.sess.run(self.generator.gan_updates,
                                 {self.generator.u: u, self.generator.i: sample, self.generator.reward: reward})

                print 'epoch:',epoch,'g_epoch:',g_epoch




    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.user[u]

            #In our experiments, discriminator performs better than gnerator
            res = self.sess.run(self.discriminator.all_rating, {self.discriminator.u: [u]})
            return res[0]

        else:
            return [self.data.globalMean] * len(self.data.item)



