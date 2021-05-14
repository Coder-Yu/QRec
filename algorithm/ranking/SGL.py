from baseclass.DeepRecommender import DeepRecommender
import tensorflow as tf
from tool import config
import numpy as np
import scipy.sparse as sp
import random

class SGL(DeepRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(SGL, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(SGL, self).readConfiguration()
        args = config.LineConfig(self.config['SGL'])
        self.reg_lambda = float(args['-lambda'])
        self.drop_rate = float(args['-droprate'])
        self.aug_type = int(args['-augtype'])

    def initModel(self):
        super(SGL, self).initModel()
        norm_adj = self._create_adj_mat(is_subgraph=False)
        ego_embeddings = tf.concat([self.user_embeddings,self.item_embeddings], axis=0)
        s1_embeddings = ego_embeddings
        s2_embeddings = ego_embeddings
        all_s1_embeddings = [s1_embeddings]
        all_s2_embeddings = [s2_embeddings]
        all_embeddings = [ego_embeddings]
        self.n_layers = 2
        #variable initialization
        self._create_variable()
        for k in range(0, self.n_layers):
            if self.aug_type in [0, 1]:
                self.sub_mat['sub_mat_1%d' % k] = tf.SparseTensor(
                    self.sub_mat['adj_indices_sub1'],
                    self.sub_mat['adj_values_sub1'],
                    self.sub_mat['adj_shape_sub1'])
                self.sub_mat['sub_mat_2%d' % k] = tf.SparseTensor(
                    self.sub_mat['adj_indices_sub2'],
                    self.sub_mat['adj_values_sub2'],
                    self.sub_mat['adj_shape_sub2'])
            else:
                self.sub_mat['sub_mat_1%d' % k] = tf.SparseTensor(
                    self.sub_mat['adj_indices_sub1%d' % k],
                    self.sub_mat['adj_values_sub1%d' % k],
                    self.sub_mat['adj_shape_sub1%d' % k])
                self.sub_mat['sub_mat_2%d' % k] = tf.SparseTensor(
                    self.sub_mat['adj_indices_sub2%d' % k],
                    self.sub_mat['adj_values_sub2%d' % k],
                    self.sub_mat['adj_shape_sub2%d' % k])
        norm_adj = self._convert_sp_mat_to_sp_tensor(norm_adj)

        #s1 - view
        for k in range(self.n_layers):
            view_embeddings = tf.sparse_tensor_dense_matmul(self.sub_mat['sub_mat_1%d' % k],s1_embeddings)
            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(view_embeddings, axis=1)
            all_s1_embeddings += [norm_embeddings]
        all_s1_embeddings = tf.reduce_sum(s1_embeddings, axis=0)
        self.s1_user_embeddings, self.s1_item_embeddings = tf.split(all_s1_embeddings, [self.num_users, self.num_items], 0)

        #s2 - view
        for k in range(self.n_layers):
            view_embeddings = tf.sparse_tensor_dense_matmul(self.sub_mat['sub_mat_2%d' % k],s2_embeddings)
            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(view_embeddings, axis=1)
            all_s2_embeddings += [norm_embeddings]
        all_s2_embeddings = tf.reduce_sum(s2_embeddings, axis=0)
        self.s2_user_embeddings, self.s2_item_embeddings = tf.split(all_s2_embeddings, [self.num_users, self.num_items], 0)

        #recommendation view
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

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _create_variable(self):
        self.sub_mat = {}
        if self.aug_type in [0, 1]:
            self.sub_mat['adj_values_sub1'] = tf.placeholder(tf.float32)
            self.sub_mat['adj_indices_sub1'] = tf.placeholder(tf.int64)
            self.sub_mat['adj_shape_sub1'] = tf.placeholder(tf.int64)

            self.sub_mat['adj_values_sub2'] = tf.placeholder(tf.float32)
            self.sub_mat['adj_indices_sub2'] = tf.placeholder(tf.int64)
            self.sub_mat['adj_shape_sub2'] = tf.placeholder(tf.int64)
        else:
            for k in range(self.n_layers):
                self.sub_mat['adj_values_sub1%d' % k] = tf.placeholder(tf.float32, name='adj_values_sub1%d' % k)
                self.sub_mat['adj_indices_sub1%d' % k] = tf.placeholder(tf.int64, name='adj_indices_sub1%d' % k)
                self.sub_mat['adj_shape_sub1%d' % k] = tf.placeholder(tf.int64, name='adj_shape_sub1%d' % k)

                self.sub_mat['adj_values_sub2%d' % k] = tf.placeholder(tf.float32, name='adj_values_sub2%d' % k)
                self.sub_mat['adj_indices_sub2%d' % k] = tf.placeholder(tf.int64, name='adj_indices_sub2%d' % k)
                self.sub_mat['adj_shape_sub2%d' % k] = tf.placeholder(tf.int64, name='adj_shape_sub2%d' % k)

    def _create_adj_mat(self, is_subgraph=False, aug_type=0):
        n_nodes = self.num_users + self.num_items
        row_idx = [self.data.user[pair[0]] for pair in self.data.trainingData]
        col_idx = [self.data.user[pair[0]] for pair in self.data.trainingData]
        if is_subgraph and aug_type in [0, 1, 2] and self.drop_rate > 0:
            # data augmentation type --- 0: Node Dropout; 1: Edge Dropout; 2: Random Walk

            if aug_type == 0:
                drop_user_idx = random.sample(range(self.num_users), size=int(self.num_users * self.drop_rate))
                drop_item_idx = random.sample(range(self.num_items), size=int(self.num_items * self.drop_rate))
                indicator_user = np.ones(self.num_users, dtype=np.float32)
                indicator_item = np.ones(self.num_items, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.
                indicator_item[drop_item_idx] = 0.
                diag_indicator_user = sp.diags(indicator_user)
                diag_indicator_item = sp.diags(indicator_item)
                R = sp.csr_matrix(
                    (np.ones_like(row_idx, dtype=np.float32), (row_idx, col_idx)),
                    shape=(self.num_users, self.num_items))
                R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
                (user_np_keep, item_np_keep) = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep+self.num_users)), shape=(n_nodes, n_nodes))
            if aug_type in [1, 2]:
                keep_idx = random.sample(range(self.data.elemCount()), size=int(len(self.data.elemCount()) * (1 - self.drop_rate)))
                user_np = np.array(row_idx)[keep_idx]
                item_np = np.array(col_idx)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.num_users)), shape=(n_nodes, n_nodes))
        else:
            user_np = np.array(row_idx)
            item_np = np.array(col_idx)
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix

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
        ssl_loss = self.reg_lambda*self.mutual_information_maximization(tf.nn.embedding_lookup(self.s1_user_embeddings,tf.unique(self.u_idx)[0]),
                                                                 tf.nn.embedding_lookup(self.s2_user_embeddings,tf.unique(self.u_idx)[0]))
        ssl_loss += self.reg_lambda*self.mutual_information_maximization(tf.nn.embedding_lookup(self.s1_item_embeddings,tf.unique(self.v_idx)[0]),
                                                                 tf.nn.embedding_lookup(self.s2_item_embeddings,tf.unique(self.v_idx)[0]))

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