from base.deepRecommender import DeepRecommender
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

class GraphRecommender(DeepRecommender):
    def __init__(self,conf,trainingSet,testSet,fold='[1]'):
        super(GraphRecommender, self).__init__(conf,trainingSet,testSet,fold)

    def create_joint_sparse_adjaceny(self):
        '''
        return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
        '''
        n_nodes = self.num_users + self.num_items
        row_idx = [self.data.user[pair[0]] for pair in self.data.trainingData]
        col_idx = [self.data.item[pair[1]] for pair in self.data.trainingData]
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

    def create_joint_sparse_adj_tensor(self):
        '''
        return a sparse tensor with the shape (user number + item number, user number + item number)
        '''
        norm_adj = self.create_joint_sparse_adjaceny()
        row,col = norm_adj.nonzero()
        indices = np.array(list(zip(row,col)))
        adj_tensor = tf.SparseTensor(indices=indices, values=norm_adj.data, dense_shape=norm_adj.shape)
        return adj_tensor

    def create_sparse_rating_matrix(self):
        '''
        return a sparse adjacency matrix with the shape (user number, item number)
        '''
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            row += [self.data.user[pair[0]]]
            col += [self.data.item[pair[1]]]
            entries += [1.0/len(self.data.trainSet_u[pair[0]])]
        ratingMat = sp.coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_items),dtype=np.float32)
        return ratingMat

    def create_sparse_adj_tensor(self):
        '''
        return a sparse tensor with the shape (user number, item number)
        '''
        ratingMat = self.create_sparse_rating_matrix()
        row,col = ratingMat.nonzero()
        indices = np.array(list(zip(row,col)))
        adj_tensor = tf.SparseTensor(indices=indices, values=ratingMat.data, dense_shape=ratingMat.shape)
        return adj_tensor