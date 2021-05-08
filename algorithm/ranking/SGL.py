from baseclass.DeepRecommender import DeepRecommender
import tensorflow as tf
from math import sqrt
from scipy.sparse import coo_matrix,csr_matrix
from tool import config
import numpy as np
import random
class SGL(DeepRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(SGL, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(SGL, self).readConfiguration()
        args = config.LineConfig(self.config['SGL'])
        self.reg_lambda = float(args['-lambda'])
        self.droprate = float(args['-droprate'])

    def buildSparseRatingMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            # symmetric matrix
            row += [self.data.user[pair[0]],self.num_users+self.data.item[pair[1]]]
            col += [self.num_users+self.data.item[pair[1]],self.data.user[pair[0]]]
            entries += [1.0,1.0]
        ratingMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users+self.num_items,self.num_users+self.num_items),dtype=np.float32)

        return ratingMatrix

    def initModel(self):
        super(SGL, self).initModel()
        ego_embeddings = tf.concat([self.user_embeddings,self.item_embeddings], axis=0)
        self.n_layers = 2

        #s1 - view
        view_embeddings = ego_embeddings
        s1_embeddings = [view_embeddings]
        self.s1_view = tf.placeholder(tf.int64)
        self.s1_values = tf.placeholder(tf.float32)
        s1_ajacency = tf.SparseTensor(indices=self.s1_view, values=self.s1_values, dense_shape=[self.num_users+self.num_items,self.num_users+self.num_items])
        for k in range(self.n_layers):
            view_embeddings = tf.sparse_tensor_dense_matmul(s1_ajacency,view_embeddings)
            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(view_embeddings, axis=1)
            s1_embeddings += [norm_embeddings]
        s1_embeddings = tf.reduce_sum(s1_embeddings, axis=0)
        self.s1_user_embeddings, self.s1_item_embeddings = tf.split(s1_embeddings, [self.num_users, self.num_items], 0)

        #s2 - view
        view_embeddings = ego_embeddings
        s2_embeddings = [view_embeddings]
        self.s2_view = tf.placeholder(tf.int64)
        self.s2_values = tf.placeholder(tf.float32)
        s2_ajacency = tf.SparseTensor(indices=self.s2_view, values=self.s2_values, dense_shape=[self.num_users + self.num_items, self.num_users + self.num_items])
        for k in range(self.n_layers):
            view_embeddings = tf.sparse_tensor_dense_matmul(s2_ajacency,view_embeddings)
            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(view_embeddings, axis=1)
            s2_embeddings += [norm_embeddings]
        s2_embeddings = tf.reduce_sum(s2_embeddings, axis=0)
        self.s2_user_embeddings, self.s2_item_embeddings = tf.split(s2_embeddings, [self.num_users, self.num_items], 0)

        #main view
        indices = [[self.data.user[item[0]],self.num_users+self.data.item[item[1]]] for item in self.data.trainingData]
        indices += [[self.num_users+self.data.item[item[1]],self.data.user[item[0]]] for item in self.data.trainingData]
        values = [float(item[2])/sqrt(len(self.data.trainSet_u[item[0]]))/sqrt(len(self.data.trainSet_i[item[1]])) for item in self.data.trainingData]*2
        norm_adj = tf.SparseTensor(indices=indices, values=values, dense_shape=[self.num_users+self.num_items,self.num_users+self.num_items])

        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            ego_embeddings = tf.sparse_tensor_dense_matmul(norm_adj,ego_embeddings)
            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]
        all_embeddings = tf.reduce_sum(all_embeddings, axis=0)

        self.main_user_embeddings, self.main_item_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)

        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_item_embedding = tf.nn.embedding_lookup(self.main_item_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(self.main_user_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.main_item_embeddings, self.v_idx)
        self.test = tf.reduce_sum(tf.multiply(self.u_embedding,self.main_item_embeddings),1)

    def edge_dropout(self):
        augmentation = [interaction for interaction in self.data.trainingData if random.random()>self.droprate]
        row = []
        col = []
        entries = []
        for pair in augmentation:
            # symmetric matrix
            row += [self.data.user[pair[0]], self.num_users + self.data.item[pair[1]]]
            col += [self.num_users + self.data.item[pair[1]], self.data.user[pair[0]]]
            entries += [1.0, 1.0]
        adjacency = coo_matrix((entries, (row, col)), shape=(self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)
        norm_adj = adjacency.multiply(1.0 / np.sqrt(adjacency.sum(axis=1)).reshape(-1, 1))
        norm_adj = norm_adj.multiply(1.0 / np.sqrt(adjacency.sum(axis=0)).reshape(1, -1))
        return np.mat(zip(np.array(row,dtype=np.int32), np.array(col,dtype=np.int32))),norm_adj.data.astype(np.float32)

    def mutual_information_maximization(self,em1,em2):
        # def row_shuffle(embedding):
        #     return tf.gather(embedding, tf.random.shuffle(tf.range(tf.shape(embedding)[0])))
        def row_column_shuffle(embedding):
            corrupted_embedding = tf.transpose(tf.gather(tf.transpose(embedding), tf.random.shuffle(tf.range(tf.shape(tf.transpose(embedding))[0]))))
            corrupted_embedding = tf.gather(corrupted_embedding, tf.random.shuffle(tf.range(tf.shape(corrupted_embedding)[0])))
            return corrupted_embedding
        def cosine(x1,x2):
            normalize_x1 = tf.nn.l2_normalize(x1, 1)
            normalize_x2 = tf.nn.l2_normalize(x2, 1)
            return tf.reduce_sum(tf.multiply(normalize_x1,normalize_x2),1)
        pos = cosine(em1,em2)
        neg = cosine(em1,row_column_shuffle(em2))
        loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos/0.2)+1e-6)-tf.log(1-tf.sigmoid(neg/0.2)+1e-6))
        return loss

    def buildModel(self):
        #main task: recommendation
        y = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1) \
            - tf.reduce_sum(tf.multiply(self.u_embedding, self.neg_item_embedding), 1)
        loss = -tf.reduce_sum(tf.log(tf.sigmoid(y))) + self.regU * (tf.nn.l2_loss(self.u_embedding) +
                                                                    tf.nn.l2_loss(self.v_embedding) +
                                                                    tf.nn.l2_loss(self.neg_item_embedding))
        #SSL task: contrastive learning
        ssl_loss = self.reg_lambda*self.mutual_information_maximization(self.s1_user_embeddings,self.s2_user_embeddings)
        ssl_loss += self.reg_lambda*self.mutual_information_maximization(self.s1_item_embeddings, self.s2_item_embeddings)

        loss+=ssl_loss
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        for iteration in range(self.maxIter):
            s1 = self.edge_dropout()
            s2 = self.edge_dropout()
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l,ssl_l = self.sess.run([train, loss, ssl_loss],
                                feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx,
                                           self.s1_view:s1[0],self.s1_values:s1[1],self.s2_view:s2[0],self.s2_values:s2[1]})
                print 'training:', iteration + 1, 'batch', n, 'loss:', l, 'ssl_loss',ssl_l

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.sess.run(self.test,feed_dict={self.u_idx:u})
        else:
            return [self.data.globalMean] * self.num_items