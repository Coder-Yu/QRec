from base.graphRecommender import GraphRecommender
from base.socialRecommender import SocialRecommender
import tensorflow as tf
from scipy.sparse import coo_matrix, eye
import scipy.sparse as sp
import numpy as np
import os
from util import config
from util.loss import bpr_loss
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#Suggested Maxium epoch LastFM: 120, Douban-Book: 30, Yelp: 30.
#Read the paper for the values of other parameters.
'''
We have transplated QRec from py2 to py3. But we found that, with py3, SEPT achieves higher NDCG
but lower (slightly) Prec and Recall compared with the results reported in the paper.
'''
class SEPT(SocialRecommender, GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)

    def readConfiguration(self):
        super(SEPT, self).readConfiguration()
        args = config.OptionConf(self.config['SEPT'])
        self.n_layers = int(args['-n_layer'])
        self.ss_rate = float(args['-ss_rate'])
        self.drop_rate = float(args['-drop_rate'])
        self.instance_cnt = int(args['-ins_cnt'])

    def buildSparseRatingMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            # symmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.item[pair[1]]]
            entries += [1.0]
        ratingMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users, self.num_items), dtype=np.float32)
        return ratingMatrix

    def get_birectional_social_matrix(self):
        row_idx = [self.data.user[pair[0]] for pair in self.social.relation]
        col_idx = [self.data.user[pair[1]] for pair in self.social.relation]
        follower_np = np.array(row_idx)
        followee_np = np.array(col_idx)
        relations = np.ones_like(follower_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((relations, (follower_np, followee_np)), shape=(self.num_users, self.num_users))
        adj_mat = tmp_adj.multiply(tmp_adj)
        return adj_mat

    def get_social_related_views(self, social_mat, rating_mat):
        def normalization(M):
            rowsum = np.array(M.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(M)
            return norm_adj_tmp.dot(d_mat_inv)
        social_matrix = social_mat.dot(social_mat)
        social_matrix =  social_matrix.multiply(social_mat) + eye(self.num_users)
        sharing_matrix = rating_mat.dot(rating_mat.T)
        sharing_matrix = sharing_matrix.multiply(social_mat) + eye(self.num_users)
        social_matrix = normalization(social_matrix)
        sharing_matrix = normalization(sharing_matrix)
        return [social_matrix, sharing_matrix]

    def _create_variable(self):
        self.sub_mat = {}
        self.sub_mat['adj_values_sub'] = tf.placeholder(tf.float32)
        self.sub_mat['adj_indices_sub'] = tf.placeholder(tf.int64)
        self.sub_mat['adj_shape_sub'] = tf.placeholder(tf.int64)
        self.sub_mat['sub_mat'] = tf.SparseTensor(
            self.sub_mat['adj_indices_sub'],
            self.sub_mat['adj_values_sub'],
            self.sub_mat['adj_shape_sub'])

    def get_adj_mat(self, is_subgraph=False):
        n_nodes = self.num_users + self.num_items
        row_idx = [self.data.user[pair[0]] for pair in self.data.trainingData]
        col_idx = [self.data.item[pair[1]] for pair in self.data.trainingData]
        s_row_idx = [self.data.user[pair[0]] for pair in self.social.relation]
        s_col_idx = [self.data.user[pair[1]] for pair in self.social.relation]
        if is_subgraph and self.drop_rate > 0:
            keep_idx = random.sample(list(range(self.data.elemCount())), int(self.data.elemCount() * (1 - self.drop_rate)))
            user_np = np.array(row_idx)[keep_idx]
            item_np = np.array(col_idx)[keep_idx]
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (user_np, self.num_users + item_np)), shape=(n_nodes, n_nodes))
            adj_mat = tmp_adj + tmp_adj.T
            skeep_idx = random.sample(list(range(len(s_row_idx))), int(len(s_row_idx) * (1 - self.drop_rate)))
            follower_np = np.array(s_row_idx)[skeep_idx]
            followee_np = np.array(s_col_idx)[skeep_idx]
            relations = np.ones_like(follower_np, dtype=np.float32)
            social_mat = sp.csr_matrix((relations, (follower_np, followee_np)), shape=(n_nodes, n_nodes))
            social_mat = social_mat.multiply(social_mat)
            adj_mat = adj_mat+social_mat
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
        super(SEPT, self).initModel()
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self._create_variable()
        self.bs_matrix = self.get_birectional_social_matrix()
        self.rating_mat = self.buildSparseRatingMatrix()
        social_mat, sharing_mat = self.get_social_related_views(self.bs_matrix, self.rating_mat)
        social_mat = self._convert_sp_mat_to_sp_tensor(social_mat)
        sharing_mat = self._convert_sp_mat_to_sp_tensor(sharing_mat)
        self.user_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.emb_size], stddev=0.005), name='U') / 2
        self.item_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.emb_size], stddev=0.005), name='V') / 2
        # initialize adjacency matrices
        ui_mat = self.create_joint_sparse_adj_tensor()
        friend_view_embeddings = self.user_embeddings
        sharing_view_embeddings = self.user_embeddings
        all_social_embeddings = [friend_view_embeddings]
        all_sharing_embeddings = [sharing_view_embeddings]
        ego_embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        all_embeddings = [ego_embeddings]
        aug_embeddings = ego_embeddings
        all_aug_embeddings = [ego_embeddings]

        #multi-view convolution: LightGCN structure
        for k in range(self.n_layers):
            # friend view
            friend_view_embeddings = tf.sparse_tensor_dense_matmul(social_mat,friend_view_embeddings)
            norm_embeddings = tf.math.l2_normalize(friend_view_embeddings, axis=1)
            all_social_embeddings += [norm_embeddings]
            # sharing view
            sharing_view_embeddings = tf.sparse_tensor_dense_matmul(sharing_mat,sharing_view_embeddings)
            norm_embeddings = tf.math.l2_normalize(sharing_view_embeddings, axis=1)
            all_sharing_embeddings += [norm_embeddings]
            # preference view
            ego_embeddings = tf.sparse_tensor_dense_matmul(ui_mat, ego_embeddings)
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]
            # unlabeled sample view
            aug_embeddings = tf.sparse_tensor_dense_matmul(self.sub_mat['sub_mat'], aug_embeddings)
            norm_embeddings = tf.math.l2_normalize(aug_embeddings, axis=1)
            all_aug_embeddings += [norm_embeddings]

        # multi-view convolution: NGCF structure
        # initializer = tf.contrib.layers.xavier_initializer()
        # self.weights = dict()
        # for k in range(self.n_layers):
        #     for view in range(4):
        #         self.weights['W_%d_1_%d' %(k,view)] = tf.Variable(
        #             initializer([self.emb_size,self.emb_size]), name='W_%d_1_%d' %(k,view))
        #         self.weights['W_%d_2_%d' %(k,view)] = tf.Variable(
        #             initializer([self.emb_size,self.emb_size]), name='W_%d_2_%d' %(k,view))
        #
        # for k in range(self.n_layers):
        #     #friend view
        #     side_embeddings = tf.sparse_tensor_dense_matmul(social_mat,friend_view_embeddings)
        #     sum_embeddings = tf.matmul(side_embeddings+friend_view_embeddings, self.weights['W_%d_1_0' % k])
        #     bi_embeddings = tf.multiply(friend_view_embeddings, side_embeddings)
        #     bi_embeddings = tf.matmul(bi_embeddings, self.weights['W_%d_2_0' % k])
        #     friend_view_embeddings = tf.nn.leaky_relu(sum_embeddings+bi_embeddings)
        #     norm_embeddings = tf.math.l2_normalize(friend_view_embeddings, axis=1)
        #     all_social_embeddings += [norm_embeddings]
        #     #sharing view
        #     side_embeddings = tf.sparse_tensor_dense_matmul(sharing_mat,sharing_view_embeddings)
        #     sum_embeddings = tf.matmul(side_embeddings+sharing_view_embeddings, self.weights['W_%d_1_1' % k])
        #     bi_embeddings = tf.multiply(sharing_view_embeddings, side_embeddings)
        #     bi_embeddings = tf.matmul(bi_embeddings, self.weights['W_%d_2_1' % k])
        #     sharing_view_embeddings = tf.nn.leaky_relu(sum_embeddings+bi_embeddings)
        #     norm_embeddings = tf.math.l2_normalize(sharing_view_embeddings, axis=1)
        #     all_sharing_embeddings += [norm_embeddings]
        #     #preference view
        #     side_embeddings = tf.sparse_tensor_dense_matmul(ui_mat, ego_embeddings)
        #     sum_embeddings = tf.matmul(side_embeddings+ego_embeddings, self.weights['W_%d_1_2' % k])
        #     bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
        #     bi_embeddings = tf.matmul(bi_embeddings, self.weights['W_%d_2_2' % k])
        #     ego_embeddings = tf.nn.leaky_relu(sum_embeddings+bi_embeddings)
        #     norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
        #     all_embeddings += [norm_embeddings]
        #     # unlabeled sample view
        #     side_embeddings = tf.sparse_tensor_dense_matmul(self.sub_mat['sub_mat'], aug_embeddings)
        #     sum_embeddings = tf.matmul(side_embeddings+aug_embeddings, self.weights['W_%d_1_3' % k])
        #     bi_embeddings = tf.multiply(aug_embeddings, side_embeddings)
        #     bi_embeddings = tf.matmul(bi_embeddings, self.weights['W_%d_2_3' % k])
        #     aug_embeddings = tf.nn.leaky_relu(sum_embeddings+bi_embeddings)
        #     norm_embeddings = tf.math.l2_normalize(aug_embeddings, axis=1)
        #     all_aug_embeddings += [norm_embeddings]

        # averaging the view-specific embeddings
        self.friend_view_embeddings = tf.reduce_sum(all_social_embeddings, axis=0)
        self.sharing_view_embeddings = tf.reduce_sum(all_sharing_embeddings, axis=0)
        all_embeddings = tf.reduce_sum(all_embeddings, axis=0)
        self.rec_user_embeddings, self.rec_item_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
        aug_embeddings = tf.reduce_sum(all_aug_embeddings, axis=0)
        self.aug_user_embeddings, self.aug_item_embeddings = tf.split(aug_embeddings, [self.num_users, self.num_items], 0)
        # embedding look-up
        self.batch_user_emb = tf.nn.embedding_lookup(self.rec_user_embeddings, self.u_idx)
        self.batch_pos_item_emb = tf.nn.embedding_lookup(self.rec_item_embeddings, self.v_idx)
        self.batch_neg_item_emb = tf.nn.embedding_lookup(self.rec_item_embeddings, self.neg_idx)

    def label_prediction(self, emb):
        emb = tf.nn.embedding_lookup(emb, tf.unique(self.u_idx)[0])
        emb = tf.nn.l2_normalize(emb, axis=1)
        aug_emb = tf.nn.embedding_lookup(self.aug_user_embeddings, tf.unique(self.u_idx)[0])
        aug_emb = tf.nn.l2_normalize(aug_emb, axis=1)
        prob = tf.matmul(emb, aug_emb, transpose_b=True)
        # avoid self-sampling
        # diag = tf.diag_part(prob)
        # prob = tf.matrix_diag(-diag)+prob
        prob = tf.nn.softmax(prob)
        return prob

    def sampling(self, logits):
        return tf.math.top_k(logits, self.instance_cnt)[1]

    def generate_pesudo_labels(self, prob1, prob2):
        positive = (prob1 + prob2) / 2
        pos_examples = self.sampling(positive)
        return pos_examples

    def neighbor_discrimination(self, positive, emb):
        def score(x1, x2):
            return tf.reduce_sum(tf.multiply(x1, x2), axis=2)
        emb = tf.nn.embedding_lookup(emb, tf.unique(self.u_idx)[0])
        emb = tf.nn.l2_normalize(emb, axis=1)
        aug_emb = tf.nn.embedding_lookup(self.aug_user_embeddings, tf.unique(self.u_idx)[0])
        aug_emb = tf.nn.l2_normalize(aug_emb, axis=1)
        pos_emb = tf.nn.embedding_lookup(aug_emb, positive)
        emb2 = tf.reshape(emb, [-1, 1, self.emb_size])
        emb2 = tf.tile(emb2, [1, self.instance_cnt, 1])
        pos = score(emb2, pos_emb)
        ttl_score = tf.matmul(emb, aug_emb, transpose_a=False, transpose_b=True)
        pos_score = tf.reduce_sum(tf.exp(pos / 0.1), axis=1)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score / 0.1), axis=1)
        ssl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))
        return ssl_loss

    def trainModel(self):
        # training the recommendation model
        rec_loss = bpr_loss(self.batch_user_emb, self.batch_pos_item_emb, self.batch_neg_item_emb)
        rec_loss += self.regU * (tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.item_embeddings))
        # self-supervision prediction
        social_prediction = self.label_prediction(self.friend_view_embeddings)
        sharing_prediction = self.label_prediction(self.sharing_view_embeddings)
        rec_prediction = self.label_prediction(self.rec_user_embeddings)
        # find informative positive examples for each encoder
        self.f_pos = self.generate_pesudo_labels(sharing_prediction, rec_prediction)
        self.sh_pos = self.generate_pesudo_labels(social_prediction, rec_prediction)
        self.r_pos = self.generate_pesudo_labels(social_prediction, sharing_prediction)
        # neighbor-discrimination based contrastive learning
        self.neighbor_dis_loss = self.neighbor_discrimination(self.f_pos, self.friend_view_embeddings)
        self.neighbor_dis_loss += self.neighbor_discrimination(self.sh_pos, self.sharing_view_embeddings)
        self.neighbor_dis_loss += self.neighbor_discrimination(self.r_pos, self.rec_user_embeddings)
        # optimizer setting
        loss = rec_loss
        loss = loss + self.ss_rate*self.neighbor_dis_loss
        v1_opt = tf.train.AdamOptimizer(self.lRate)
        v1_op = v1_opt.minimize(rec_loss)
        v2_opt = tf.train.AdamOptimizer(self.lRate)
        v2_op = v2_opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.maxEpoch):
            #joint learning
            if epoch > self.maxEpoch / 3:
                sub_mat = {}
                sub_mat['adj_indices_sub'], sub_mat['adj_values_sub'], sub_mat[
                    'adj_shape_sub'] = self._convert_csr_to_sparse_tensor_inputs(
                    self.get_adj_mat(is_subgraph=True))
                for n, batch in enumerate(self.next_batch_pairwise()):
                    user_idx, i_idx, j_idx = batch
                    feed_dict = {self.u_idx: user_idx,
                                 self.v_idx: i_idx,
                                 self.neg_idx: j_idx}
                    feed_dict.update({
                        self.sub_mat['adj_values_sub']: sub_mat['adj_values_sub'],
                        self.sub_mat['adj_indices_sub']: sub_mat['adj_indices_sub'],
                        self.sub_mat['adj_shape_sub']: sub_mat['adj_shape_sub'],
                    })
                    _, l1, l3, = self.sess.run([v2_op, rec_loss, self.neighbor_dis_loss],feed_dict=feed_dict)
                    print(self.foldInfo, 'training:', epoch + 1, 'batch', n, 'rec loss:', l1, 'con_loss:', self.ss_rate*l3)
            else:
                #initialization with only recommendation task
                for n, batch in enumerate(self.next_batch_pairwise()):
                    user_idx, i_idx, j_idx = batch
                    feed_dict = {self.u_idx: user_idx,
                                 self.v_idx: i_idx,
                                 self.neg_idx: j_idx}
                    _, l1 = self.sess.run([v1_op, rec_loss],
                                          feed_dict=feed_dict)
                    print(self.foldInfo, 'training:', epoch + 1, 'batch', n, 'rec loss:', l1)
            self.U, self.V = self.sess.run([self.rec_user_embeddings, self.rec_item_embeddings])
            self.ranking_performance(epoch)
        self.U,self.V = self.bestU,self.bestV

    def saveModel(self):
        self.bestU, self.bestV = self.sess.run([self.rec_user_embeddings, self.rec_item_embeddings])

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items