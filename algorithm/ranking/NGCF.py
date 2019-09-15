#coding:utf8
from baseclass.DeepRecommender import DeepRecommender
from random import choice
import tensorflow as tf
import numpy as np
from math import sqrt
class NGCF(DeepRecommender):

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(NGCF, self).__init__(conf,trainingSet,testSet,fold)

    def next_batch(self):
        batch_id = 0
        while batch_id < self.train_size:
            if batch_id + self.batch_size <= self.train_size:
                users = [self.data.trainingData[idx][0] for idx in range(batch_id, self.batch_size + batch_id)]
                items = [self.data.trainingData[idx][1] for idx in range(batch_id, self.batch_size + batch_id)]
                batch_id += self.batch_size
            else:
                users = [self.data.trainingData[idx][0] for idx in range(batch_id, self.train_size)]
                items = [self.data.trainingData[idx][1] for idx in range(batch_id, self.train_size)]
                batch_id = self.train_size

            u_idx, i_idx, j_idx = [], [], []
            item_list = self.data.item.keys()
            for i, user in enumerate(users):

                i_idx.append(self.data.item[items[i]])
                u_idx.append(self.data.user[user])

                neg_item = choice(item_list)
                while neg_item in self.data.trainSet_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(self.data.item[neg_item])

            yield u_idx, i_idx, j_idx

    def initModel(self):
        super(NGCF, self).initModel()

        ego_embeddings = tf.concat([self.user_embeddings,self.item_embeddings], axis=0)

        all_embeddings = [ego_embeddings]

        indices = [[self.data.user[item[0]],self.num_users+self.data.item[item[1]]] for item in self.data.trainingData]
        indices += [self.num_users+[self.data.item[item[1]],self.data.user[item[0]]] for item in self.data.trainingData]
        values = [self.data.trainingData[item[2]/sqrt(len(self.data.trainSet_u[item[0]]))/
                                         sqrt(len(self.data.trainSet_i[item[1]]))] for item in self.data.trainingData]*2

        norm_adj = tf.SparseTensor(indices=indices, values=values, dense_shape=[self.num_users+self.num_items,self.num_items+self.num_items])



        # self.u_neighbors_matrix = tf.placeholder(tf.int32, [None, self.num_items], name="u_n_idx")
        # self.i_Neighbors_matrix = tf.placeholder(tf.int32, [None, self.num_users], name="i_n_idx")
        # self.j_idx = tf.placeholder(tf.int32, [None], name="j_idx")
        # self.p_u = tf.placeholder(tf.int32, [None], name="j_idx")
        # self.p_i = tf.placeholder(tf.int32, [None], name="j_idx")
        # self.p_j = tf.placeholder(tf.int32, [None], name="j_idx")
        #
        # decay_u = np.zeros(self.num_users)
        #
        # for user in self.data.user:
        #     uid = self.data.user[user]
        #     decay_u[uid] = sqrt(len(self.data.trainSet_u[user]))
        # decay_u = tf.convert_to_tensor(decay_u)
        # decay_i = np.zeros(self.num_items)
        #
        # for item in self.data.item:
        #     iid = self.data.user[item]
        #     decay_i[iid] = sqrt(len(self.data.trainSet_i[item]))
        # decay_i = tf.convert_to_tensor(decay_i)

        self.weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()
        weight_size = [self.embed_size*4,self.embed_size*2,self.embed_size]
        weight_size_list = [self.embed_size] + weight_size

        self.n_layers = 3

        #initialize parameters
        for k in range(self.n_layers):
            self.weights['W_%d_1' % k] = tf.Variable(
                initializer([weight_size_list[k], weight_size_list[k + 1]]), name='W_%d_1' % k)
            self.weights['W_%d_2' % k] = tf.Variable(
                initializer([weight_size_list[k], weight_size_list[k + 1]]), name='W_%d_2' % k)

        for k in range(self.n_layers):



        # for k in range(-1,self.n_layers):
        #     self.variables['user_embeddings_%d' % k] = tf.Variable(
        #         tf.truncated_normal(shape=[self.num_users, self.embed_size], stddev=0.005),
        #         name='user_embeddings_d' % k)
        #     self.variables['item_embeddings_%d' % k] = tf.Variable(
        #         tf.truncated_normal(shape=[self.num_items, self.embed_size], stddev=0.005),
        #         name='user_embeddings_d' % k)
        #     self.variables['u_embedding_%d'] = tf.nn.embedding_lookup(self.variables['user_embeddings_%d' % k],
        #                                                               self.u_idx)
        #     self.variables['v_embedding_%d'] = tf.nn.embedding_lookup(self.variables['item_embeddings_%d' % k],
        #                                                               self.v_idx)
        #     self.variables['j_embedding_%d'] = tf.nn.embedding_lookup(self.variables['item_embeddings_%d' % k],
        #                                                               self.j_idx)

        # self.neighbors_u = tf.Placeholder(tf.int32,[None,self.num_items])
        # self.neighbors_v = tf.Placeholder(tf.int32,[None,self.num_users])
        # self.neighbors_j = tf.Placeholder(tf.int32,[None,self.num_users])

        #all_embeddings =
        # for k in range(0,self.n_layers):
        #
        #     # aggregate messages of items.
        #     sum_item_messages = tf.matmul(self.neighbors_u,self.variables['item_embeddings_%d' %(k-1)]/decay_i)
        #     W_1_e_i = tf.matmul(self.variables['W_%d_1' % k],sum_item_messages,transpose_b=True)
        #     sum_item_messages = tf.multiply(self.variables['u_embedding_%d' %(k-1)]/self.p_u,sum_item_messages)
        #     sum_item_messages = tf.matmul(self.variables['W_%d_2' % k],sum_item_messages,transpose_b=True)
        #     sum_item_messages += W_1_e_i
        #     e_u = tf.nn.leaky_relu(tf.matmul(self.variables['W_%d_1' % k],self.variables['u_embedding_%d' %(k-1)],
        #                                      transpose_b=True)+sum_item_messages)
        #
        #     # aggregate messages of positive item.
        #     sum_user_messages = tf.matmul(self.neighbors_v, self.variables['user_embeddings_%d' %(k-1)] / decay_u)
        #     W_1_e_u = tf.matmul(self.variables['W_%d_1' % k], sum_user_messages, transpose_b=True)
        #     sum_user_messages = tf.multiply(self.variables['v_embedding_%d' %(k-1)] / self.p_i, sum_user_messages)
        #     sum_user_messages = tf.matmul(self.variables['W_%d_2' % k], sum_user_messages, transpose_b=True)
        #     sum_user_messages += W_1_e_u
        #     e_i = tf.nn.leaky_relu(tf.matmul(self.variables['W_%d_1' % k], self.variables['v_embedding_%d' %(k-1)],
        #                                      transpose_b=True) + sum_user_messages)
        #
        #     # aggregate messages of negative item.
        #     sum_user_messages = tf.matmul(self.neighbors_j, self.variables['user_embeddings_%d' % %(k-1)] / decay_u)
        #     W_1_e_u = tf.matmul(self.variables['W_%d_1' % k], sum_user_messages, transpose_b=True)
        #     sum_user_messages = tf.multiply(self.variables['j_embedding_%d' %(k-1)] / self.p_j, sum_user_messages)
        #     sum_user_messages = tf.matmul(self.variables['W_%d_2' % k], sum_user_messages, transpose_b=True)
        #     sum_user_messages += W_1_e_u
        #     e_j = tf.nn.leaky_relu(tf.matmul(self.variables['W_%d_1' % k], self.variables['j_embedding_%d' %(k-1)],
        #                                      transpose_b=True) + sum_user_messages)




            # # message dropout.
            # ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.prob_dropout[k])
            #
            # # normalize the distribution of embeddings.
            # norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)



    def buildModel(self):

        init = tf.global_variables_initializer()
        self.sess.run(init)

        print 'training... (NeuMF)'
        for iteration in range(self.maxIter/5):
            for num, batch in enumerate(self.next_batch()):
                user_idx, item_idx, r = batch
                _, loss, y_neu = self.sess.run([self.neu_optimizer, self.neu_loss, self.y_neu],
                                          feed_dict={self.u_idx: user_idx, self.i_idx: item_idx, self.r: r})
                print 'iteration:', iteration, 'batch:', num, 'loss:', loss


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