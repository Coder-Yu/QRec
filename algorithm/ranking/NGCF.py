#coding:utf8
from baseclass.DeepRecommender import DeepRecommender
from random import choice
import tensorflow as tf

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

    def getNeignbors(self):
        user_Neighbors = dict()
        item_Neighbors = dict()
        for user in self.data.user:
            uid = self.data.user[user]
            user_Neighbors[uid]=[self.data.item[item] for item in self.data.trainSet_u[user]]
        for item in self.data.item:
            iid = self.data.item[item]
            item_Neighbors[iid]=[self.data.user[user] for user in self.data.trainSet_i[item]]

        return user_Neighbors,item_Neighbors


    def initModel(self):
        super(NGCF, self).initModel()

        self.u_neighbors_matrix = tf.placeholder(tf.int32, [None, self.num_items], name="u_n_idx")
        self.i_Neighbors_matrix = tf.placeholder(tf.int32, [None, self.num_users], name="i_n_idx")


        self.u_ = tf.nn.embedding_lookup(self.user_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.v_idx)

        self.weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()
        self.weight_size = [self.embed_size*4,self.embed_size*2,self.embed_size]
        self.weight_size_list = [self.embed_size] + self.weight_size
        self.n_layers = 3

        for k in range(self.n_layers):
            self.weights['W_%d_1' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_%d_1' % k)
            self.weights['b_%d_2' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_%d_2' % k)





        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):


            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # transformed sum messages of neighbors.
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])

            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.prob_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.num_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def buildModel(self):

        init = tf.global_variables_initializer()
        self.sess.run(init)

        print 'pretraining... (GMF)'
        for iteration in range(self.maxIter):
            for num,batch in enumerate(self.next_batch()):
                user_idx, item_idx, r = batch

                _, loss,y_mf = self.sess.run([self.mf_optimizer, self.mf_loss,self.y_mf],
                                   feed_dict={self.u_idx: user_idx, self.i_idx: item_idx, self.r: r})
                print 'iteration:', iteration, 'batch:', num, 'loss:', loss

        print 'pretraining... (MLP)'
        for iteration in range(self.maxIter/2):
            for num, batch in enumerate(self.next_batch()):
                user_idx, item_idx, r = batch
                _, loss, y_mlp = self.sess.run([self.mlp_optimizer, self.mlp_loss, self.y_mlp],
                                          feed_dict={self.u_idx: user_idx, self.i_idx: item_idx, self.r: r})
                print 'iteration:', iteration, 'batch:', num, 'loss:', loss

        print 'training... (NeuMF)'
        for iteration in range(self.maxIter/5):
            for num, batch in enumerate(self.next_batch()):
                user_idx, item_idx, r = batch
                _, loss, y_neu = self.sess.run([self.neu_optimizer, self.neu_loss, self.y_neu],
                                          feed_dict={self.u_idx: user_idx, self.i_idx: item_idx, self.r: r})
                print 'iteration:', iteration, 'batch:', num, 'loss:', loss

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