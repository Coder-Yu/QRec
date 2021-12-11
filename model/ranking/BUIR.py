from base.deepRecommender import DeepRecommender
from base.socialRecommender import SocialRecommender
import tensorflow as tf
from scipy.sparse import coo_matrix, eye
import scipy.sparse as sp
import numpy as np
import os
from util import config
import random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#Suggested Maxium epoch LastFM: 120, Douban-Book: 30, Yelp: 30.
#Read the paper for the values of other parameters.
class BUIR(DeepRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, fold='[1]'):
        DeepRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)

    def readConfiguration(self):
        super(BUIR, self).readConfiguration()
        args = config.OptionConf(self.config['BUIR'])
        self.n_layers = int(args['-n_layer'])
        self.tau = float(args['-tau'])
        self.drop_rate = float(args['-drop_rate'])

    def _create_variable(self):
        self.sub_mat = {}
        self.sub_mat['adj_values_sub_o'] = tf.placeholder(tf.float32)
        self.sub_mat['adj_indices_sub_o'] = tf.placeholder(tf.int64)
        self.sub_mat['adj_shape_sub_o'] = tf.placeholder(tf.int64)
        self.sub_mat['sub_mat_o'] = tf.SparseTensor(
            self.sub_mat['adj_indices_sub_o'],
            self.sub_mat['adj_values_sub_o'],
            self.sub_mat['adj_shape_sub_o'])
        self.sub_mat['adj_values_sub_t'] = tf.placeholder(tf.float32)
        self.sub_mat['adj_indices_sub_t'] = tf.placeholder(tf.int64)
        self.sub_mat['adj_shape_sub_t'] = tf.placeholder(tf.int64)
        self.sub_mat['sub_mat_t'] = tf.SparseTensor(
            self.sub_mat['adj_indices_sub_t'],
            self.sub_mat['adj_values_sub_t'],
            self.sub_mat['adj_shape_sub_t'])

    def get_adj_mat(self, is_subgraph=False):
        n_nodes = self.num_users + self.num_items
        row_idx = [self.data.user[pair[0]] for pair in self.data.trainingData]
        col_idx = [self.data.item[pair[1]] for pair in self.data.trainingData]
        if is_subgraph and self.drop_rate > 0:
            keep_idx = random.sample(list(range(self.data.elemCount())), int(self.data.elemCount() * (1 - self.drop_rate)))
            user_np = np.array(row_idx)[keep_idx]
            item_np = np.array(col_idx)[keep_idx]
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (user_np, self.num_users + item_np)), shape=(n_nodes, n_nodes))
            adj_mat = tmp_adj + tmp_adj.T
        else:
            user_np = np.array(row_idx)
            item_np = np.array(col_idx)
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))
            adj_mat = tmp_adj + tmp_adj.T
        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _convert_csr_to_sparse_tensor_inputs(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape

    def initModel(self):
        super(BUIR, self).initModel()
        self._create_variable()
        initializer = tf.contrib.layers.xavier_initializer()
        self.online_mat = tf.Variable(initializer([self.emb_size, self.emb_size]))
        self.online_bias = tf.Variable(initializer([1, self.emb_size]))
        self.online_user_embeddings = tf.Variable(initializer(shape=[self.num_users, self.emb_size]), name='U')
        self.online_item_embeddings = tf.Variable(initializer(shape=[self.num_items, self.emb_size]), name='V')
        self.target_user_embeddings = tf.Variable(self.online_user_embeddings.initialized_value(), name='t_U')
        self.target_item_embeddings = tf.Variable(self.online_item_embeddings.initialized_value(), name='t_V')
        # initialize adjacency matrices
        online_embeddings = tf.concat([self.online_user_embeddings, self.online_item_embeddings], axis=0)
        target_embeddings = tf.concat([self.target_user_embeddings, self.target_item_embeddings], axis=0)
        all_online_embeddings = [online_embeddings]
        all_target_embeddings = [target_embeddings]
        #multi-view convolution: LightGCN structure
        for k in range(self.n_layers):
            # online encoder
            online_embeddings = tf.sparse_tensor_dense_matmul(self.sub_mat['sub_mat_o'], online_embeddings)
            #norm_embeddings = tf.math.l2_normalize(online_embeddings, axis=1)
            all_online_embeddings += [online_embeddings]
            # target encoder
            target_embeddings = tf.sparse_tensor_dense_matmul(self.sub_mat['sub_mat_t'], target_embeddings)
            #norm_embeddings = tf.math.l2_normalize(target_embeddings, axis=1)
            all_target_embeddings += [target_embeddings]
        # averaging the view-specific embeddings
        online_embeddings = tf.reduce_mean(all_online_embeddings, axis=0)
        self.on_user_embeddings, self.on_item_embeddings = tf.split(online_embeddings, [self.num_users, self.num_items], 0)
        #linear layer
        def linear(em):
            return tf.nn.tanh(tf.matmul(em,self.online_mat)+self.online_bias)
        q_online_embeddings = linear(online_embeddings)
        self.q_user_embeddings, self.q_item_embeddings = tf.split(q_online_embeddings, [self.num_users, self.num_items], 0)
        target_embeddings = tf.reduce_mean(all_target_embeddings, axis=0)
        tar_user_embeddings, tar_item_embeddings = tf.split(target_embeddings, [self.num_users, self.num_items], 0)
        tar_user_embeddings = tf.stop_gradient(tar_user_embeddings)
        tar_item_embeddings = tf.stop_gradient(tar_item_embeddings)
        # embedding look-up
        self.q_u_embedding = tf.nn.embedding_lookup(self.q_user_embeddings, self.u_idx)
        self.q_i_embedding = tf.nn.embedding_lookup(self.q_item_embeddings, self.v_idx)
        self.u_tar_embedding = tf.nn.embedding_lookup(tar_user_embeddings, self.u_idx)
        self.i_tar_embedding = tf.nn.embedding_lookup(tar_item_embeddings, self.v_idx)
        #target_update
        momentum_u = self.target_user_embeddings*self.tau + self.online_user_embeddings*(1 - self.tau)
        momentum_i = self.target_item_embeddings*self.tau + self.online_item_embeddings*(1 - self.tau)
        self.tar_user_update = self.target_user_embeddings.assign(momentum_u)
        self.tar_item_update = self.target_item_embeddings.assign(momentum_i)
        #test
        self.pred = tf.reduce_sum(tf.multiply(self.q_u_embedding,self.on_item_embeddings),1)\
                    +tf.reduce_sum(tf.multiply(tf.nn.embedding_lookup(self.on_user_embeddings,self.u_idx),self.q_item_embeddings),1)
    def trainModel(self):
        # computing loss
        loss = 1-tf.reduce_sum(tf.multiply(tf.math.l2_normalize(self.q_u_embedding,axis=1),tf.math.l2_normalize(self.i_tar_embedding,axis=1)),1)
        loss +=1-tf.reduce_sum(tf.multiply(tf.math.l2_normalize(self.q_i_embedding,axis=1),tf.math.l2_normalize(self.u_tar_embedding,axis=1)),1)
        loss = tf.reduce_sum(loss/2)#+self.regU * (tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.item_embeddings)
                                                 #+tf.nn.l2_loss(self.online_mat)+tf.nn.l2_loss(self.online_bias))
        # optimizer setting
        learner = tf.train.AdamOptimizer(self.lRate)
        op = learner.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        #training
        for epoch in range(self.maxEpoch):
            sub_mat = {}
            sub_mat['adj_indices_sub_o'], sub_mat['adj_values_sub_o'], sub_mat[
                'adj_shape_sub_o'] = self._convert_csr_to_sparse_tensor_inputs(
                self.get_adj_mat(is_subgraph=True))
            sub_mat['adj_indices_sub_t'], sub_mat['adj_values_sub_t'], sub_mat[
                'adj_shape_sub_t'] = self._convert_csr_to_sparse_tensor_inputs(
                self.get_adj_mat(is_subgraph=True))
            for iteration, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                feed_dict = {self.u_idx: user_idx,
                             self.v_idx: i_idx}
                feed_dict.update({
                    self.sub_mat['adj_values_sub_o']: sub_mat['adj_values_sub_o'],
                    self.sub_mat['adj_indices_sub_o']: sub_mat['adj_indices_sub_o'],
                    self.sub_mat['adj_shape_sub_o']: sub_mat['adj_shape_sub_o'],
                })
                feed_dict.update({
                    self.sub_mat['adj_values_sub_t']: sub_mat['adj_values_sub_t'],
                    self.sub_mat['adj_indices_sub_t']: sub_mat['adj_indices_sub_t'],
                    self.sub_mat['adj_shape_sub_t']: sub_mat['adj_shape_sub_t'],
                })
                _, l1,  = self.sess.run([op, loss],feed_dict=feed_dict)
                print(self.foldInfo, 'training:', epoch + 1, 'batch', iteration, 'loss:', l1)
                self.sess.run([self.tar_user_update, self.tar_item_update])
        self.final_sub_mat = {}
        self.final_sub_mat['adj_indices_sub_o'], self.final_sub_mat['adj_values_sub_o'], self.final_sub_mat[
            'adj_shape_sub_o'] = self._convert_csr_to_sparse_tensor_inputs(self.get_adj_mat())
        self.q_user, self.q_item, self.o_user, self.o_item = self.sess.run([self.q_user_embeddings,self.q_item_embeddings,self.on_user_embeddings,self.on_item_embeddings],feed_dict={
                        self.sub_mat['adj_values_sub_o']: self.final_sub_mat['adj_values_sub_o'],
                        self.sub_mat['adj_indices_sub_o']: self.final_sub_mat['adj_indices_sub_o'],
                        self.sub_mat['adj_shape_sub_o']: self.final_sub_mat['adj_shape_sub_o'],})

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.q_item.dot(self.o_user[u])+self.o_item.dot(self.q_user[u])
        else:
            return [self.data.globalMean] * self.num_items