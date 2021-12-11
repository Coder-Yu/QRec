from base.graphRecommender import GraphRecommender
import tensorflow as tf
from util import config
from util.loss import bpr_loss
import numpy as np
import scipy.sparse as sp
import random
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
class SGL(GraphRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(SGL, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(SGL, self).readConfiguration()
        args = config.OptionConf(self.config['SGL'])
        self.ssl_reg = float(args['-lambda'])
        self.drop_rate = float(args['-droprate'])
        self.aug_type = int(args['-augtype'])
        self.ssl_temp = float(args['-temp'])
        self.n_layers = int(args['-n_layer'])

    def initModel(self):
        super(SGL, self).initModel()
        norm_adj = self._create_adj_mat(is_subgraph=False)
        norm_adj = self._convert_sp_mat_to_sp_tensor(norm_adj)
        ego_embeddings = tf.concat([self.user_embeddings,self.item_embeddings], axis=0)
        s1_embeddings = ego_embeddings
        s2_embeddings = ego_embeddings
        all_s1_embeddings = [s1_embeddings]
        all_s2_embeddings = [s2_embeddings]
        all_embeddings = [ego_embeddings]
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

        #s1 - view
        for k in range(self.n_layers):
            s1_embeddings = tf.sparse_tensor_dense_matmul(self.sub_mat['sub_mat_1%d' % k],s1_embeddings)
            all_s1_embeddings += [s1_embeddings]
        all_s1_embeddings = tf.stack(all_s1_embeddings, 1)
        all_s1_embeddings = tf.reduce_mean(all_s1_embeddings, axis=1, keepdims=False)
        self.s1_user_embeddings, self.s1_item_embeddings = tf.split(all_s1_embeddings, [self.num_users, self.num_items], 0)

        #s2 - view
        for k in range(self.n_layers):
            s2_embeddings = tf.sparse_tensor_dense_matmul(self.sub_mat['sub_mat_2%d' % k],s2_embeddings)
            all_s2_embeddings += [s2_embeddings]
        all_s2_embeddings = tf.stack(all_s2_embeddings, 1)
        all_s2_embeddings = tf.reduce_mean(all_s2_embeddings, axis=1, keepdims=False)
        self.s2_user_embeddings, self.s2_item_embeddings = tf.split(all_s2_embeddings, [self.num_users, self.num_items], 0)
        #recommendation view
        for k in range(self.n_layers):
            ego_embeddings = tf.sparse_tensor_dense_matmul(norm_adj,ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        self.main_user_embeddings, self.main_item_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.batch_neg_item_emb = tf.nn.embedding_lookup(self.main_item_embeddings, self.neg_idx)
        self.batch_user_emb = tf.nn.embedding_lookup(self.main_user_embeddings, self.u_idx)
        self.batch_pos_item_emb = tf.nn.embedding_lookup(self.main_item_embeddings, self.v_idx)


    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _convert_csr_to_sparse_tensor_inputs(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape

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
        col_idx = [self.data.item[pair[1]] for pair in self.data.trainingData]
        if is_subgraph and aug_type in [0, 1, 2] and self.drop_rate > 0:
            # data augmentation type --- 0: Node Dropout; 1: Edge Dropout; 2: Random Walk
            if aug_type == 0:
                drop_user_idx = random.sample(list(range(self.num_users)), int(self.num_users * self.drop_rate))
                drop_item_idx = random.sample(list(range(self.num_items)), int(self.num_items * self.drop_rate))
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
                keep_idx = random.sample(list(range(self.data.elemCount())), int(self.data.elemCount() * (1 - self.drop_rate)))
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

    def calc_ssl_loss(self):
        '''
        Calculating SSL loss
        '''

        user_emb1 = tf.nn.embedding_lookup(self.s1_user_embeddings, tf.unique(self.u_idx)[0])
        user_emb2 = tf.nn.embedding_lookup(self.s2_user_embeddings, tf.unique(self.u_idx)[0])
        normalize_user_emb1 = tf.nn.l2_normalize(user_emb1, 1)
        normalize_user_emb2 = tf.nn.l2_normalize(user_emb2, 1)

        item_emb1 = tf.nn.embedding_lookup(self.s1_item_embeddings, tf.unique(self.v_idx)[0])
        item_emb2 = tf.nn.embedding_lookup(self.s2_item_embeddings, tf.unique(self.v_idx)[0])
        normalize_item_emb1 = tf.nn.l2_normalize(item_emb1, 1)
        normalize_item_emb2 = tf.nn.l2_normalize(item_emb2, 1)

        normalize_user_emb2_neg = normalize_user_emb2
        normalize_item_emb2_neg = normalize_item_emb2

        pos_score_user = tf.reduce_sum(tf.multiply(normalize_user_emb1, normalize_user_emb2), axis=1)
        ttl_score_user = tf.matmul(normalize_user_emb1, normalize_user_emb2_neg, transpose_a=False, transpose_b=True)

        pos_score_item = tf.reduce_sum(tf.multiply(normalize_item_emb1, normalize_item_emb2), axis=1)
        ttl_score_item = tf.matmul(normalize_item_emb1, normalize_item_emb2_neg, transpose_a=False, transpose_b=True)

        pos_score_user = tf.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = tf.reduce_sum(tf.exp(ttl_score_user / self.ssl_temp), axis=1)
        pos_score_item = tf.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = tf.reduce_sum(tf.exp(ttl_score_item / self.ssl_temp), axis=1)

        # ssl_loss = -tf.reduce_mean(tf.log(pos_score / ttl_score))
        ssl_loss_user = -tf.reduce_sum(tf.log(pos_score_user / ttl_score_user))
        ssl_loss_item = -tf.reduce_sum(tf.log(pos_score_item / ttl_score_item))
        ssl_loss = self.ssl_reg*(ssl_loss_user + ssl_loss_item)
        return ssl_loss

    def calc_ssl_loss_v3(self):
        '''
        The denominator is summation over the user and item examples in a batch
        '''

        user_emb1 = tf.nn.embedding_lookup(self.s1_user_embeddings, tf.unique(self.u_idx)[0])
        user_emb2 = tf.nn.embedding_lookup(self.s2_user_embeddings, tf.unique(self.u_idx)[0])

        item_emb1 = tf.nn.embedding_lookup(self.s1_item_embeddings, tf.unique(self.v_idx)[0])
        item_emb2 = tf.nn.embedding_lookup(self.s2_item_embeddings, tf.unique(self.v_idx)[0])

        emb_merge1 = tf.concat([user_emb1, item_emb1], axis=0)
        emb_merge2 = tf.concat([user_emb2, item_emb2], axis=0)

        # cosine similarity
        normalize_emb_merge1 = tf.nn.l2_normalize(emb_merge1, 1)
        normalize_emb_merge2 = tf.nn.l2_normalize(emb_merge2, 1)

        pos_score = tf.reduce_sum(tf.multiply(normalize_emb_merge1, normalize_emb_merge2), axis=1)
        ttl_score = tf.matmul(normalize_emb_merge1, normalize_emb_merge2, transpose_a=False, transpose_b=True)

        pos_score = tf.exp(pos_score / self.ssl_temp)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score / self.ssl_temp), axis=1)
        ssl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))
        ssl_loss = self.ssl_reg * ssl_loss
        return ssl_loss

    def trainModel(self):
        #main task: recommendation
        rec_loss = bpr_loss(self.batch_user_emb,self.batch_pos_item_emb,self.batch_neg_item_emb)
        rec_loss +=  self.regU * (tf.nn.l2_loss(self.batch_user_emb) + tf.nn.l2_loss(self.batch_pos_item_emb) + tf.nn.l2_loss(self.batch_neg_item_emb))
        #SSL task: contrastive learning
        ssl_loss = self.calc_ssl_loss_v3()
        total_loss = rec_loss+ssl_loss

        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(total_loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        import time
        for epoch in range(self.maxEpoch):
            sub_mat = {}
            if self.aug_type in [0, 1]:
                sub_mat['adj_indices_sub1'], sub_mat['adj_values_sub1'], sub_mat[
                    'adj_shape_sub1'] = self._convert_csr_to_sparse_tensor_inputs(
                    self._create_adj_mat(is_subgraph=True, aug_type=self.aug_type))

                sub_mat['adj_indices_sub2'], sub_mat['adj_values_sub2'], sub_mat[
                    'adj_shape_sub2'] = self._convert_csr_to_sparse_tensor_inputs(
                    self._create_adj_mat(is_subgraph=True, aug_type=self.aug_type))
            else:
                for k in range(self.n_layers):
                    sub_mat['adj_indices_sub1%d' % k], sub_mat['adj_values_sub1%d' % k], sub_mat[
                        'adj_shape_sub1%d' % k] = self._convert_csr_to_sparse_tensor_inputs(
                        self._create_adj_mat(is_subgraph=True, aug_type=self.aug_type))
                    sub_mat['adj_indices_sub2%d' % k], sub_mat['adj_values_sub2%d' % k], sub_mat[
                        'adj_shape_sub2%d' % k] = self._convert_csr_to_sparse_tensor_inputs(
                        self._create_adj_mat(is_subgraph=True, aug_type=self.aug_type))
            s = time.time()
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                feed_dict = {self.u_idx: user_idx,
                             self.v_idx: i_idx,
                             self.neg_idx: j_idx, }
                if self.aug_type in [0, 1]:
                    feed_dict.update({
                        self.sub_mat['adj_values_sub1']: sub_mat['adj_values_sub1'],
                        self.sub_mat['adj_indices_sub1']: sub_mat['adj_indices_sub1'],
                        self.sub_mat['adj_shape_sub1']: sub_mat['adj_shape_sub1'],
                        self.sub_mat['adj_values_sub2']: sub_mat['adj_values_sub2'],
                        self.sub_mat['adj_indices_sub2']: sub_mat['adj_indices_sub2'],
                        self.sub_mat['adj_shape_sub2']: sub_mat['adj_shape_sub2']
                    })
                else:
                    for k in range(self.n_layers):
                        feed_dict.update({
                            self.sub_mat['adj_values_sub1%d' % k]: sub_mat['adj_values_sub1%d' % k],
                            self.sub_mat['adj_indices_sub1%d' % k]: sub_mat['adj_indices_sub1%d' % k],
                            self.sub_mat['adj_shape_sub1%d' % k]: sub_mat['adj_shape_sub1%d' % k],
                            self.sub_mat['adj_values_sub2%d' % k]: sub_mat['adj_values_sub2%d' % k],
                            self.sub_mat['adj_indices_sub2%d' % k]: sub_mat['adj_indices_sub2%d' % k],
                            self.sub_mat['adj_shape_sub2%d' % k]: sub_mat['adj_shape_sub2%d' % k]
                        })

                _, l,rec_l,ssl_l = self.sess.run([train, total_loss, rec_loss, ssl_loss],feed_dict=feed_dict)
                print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_l, 'ssl_loss',ssl_l)
            self.U, self.V = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])
            self.ranking_performance(epoch)
        self.U, self.V = self.bestU, self.bestV


    def saveModel(self):
        self.bestU, self.bestV = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])

    def predictForRanking(self, u):
        'rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items
