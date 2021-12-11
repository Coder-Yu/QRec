#coding:utf8
from base.deepRecommender import DeepRecommender
from scipy.sparse import coo_matrix,hstack
import tensorflow as tf
import numpy as np

#The original implementation is not opensourced. I reproduced this model by emailing with the author.
#However, I think this model is not effective. There are a lot of problems in the original paper.
#Build 2-hop hyperedge would lead to a very dense adjacency matrix. I think it would result in over-smoothing.
#So, I just use the 1-hop hyperedge.

class DHCF(DeepRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(DHCF, self).__init__(conf,trainingSet,testSet,fold)

    def buildAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            # symmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.item[pair[1]]]
            entries += [1]
        u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_items),dtype=np.float32)
        return u_i_adj

    def initModel(self):
        super(DHCF, self).initModel()
        #Build adjacency matrix
        A = self.buildAdjacencyMatrix()
        #Build incidence matrix
        #H_u = hstack([A,A.dot(A.transpose().dot(A))])
        H_u = A
        D_u_v = H_u.sum(axis=1).reshape(1,-1)
        D_u_e = H_u.sum(axis=0).reshape(1,-1)
        temp1 = (H_u.transpose().multiply(np.sqrt(1.0/D_u_v))).transpose()
        temp2 = temp1.transpose()
        A_u = temp1.multiply(1.0/D_u_e).dot(temp2)
        A_u = A_u.tocoo()
        indices = np.mat([A_u.row, A_u.col]).transpose()
        H_u = tf.SparseTensor(indices, A_u.data.astype(np.float32), A_u.shape)

        H_i = A.transpose()
        D_i_v = H_i.sum(axis=1).reshape(1,-1)
        D_i_e = H_i.sum(axis=0).reshape(1,-1)
        temp1 = (H_i.transpose().multiply(np.sqrt(1.0 / D_i_v))).transpose()
        temp2 = temp1.transpose()
        A_i = temp1.multiply(1.0 / D_i_e).dot(temp2)
        A_i = A_i.tocoo()
        indices = np.mat([A_i.row, A_i.col]).transpose()
        H_i = tf.SparseTensor(indices, A_i.data.astype(np.float32), A_i.shape)

        print('Runing on GPU...')
        #Build network
        self.isTraining = tf.placeholder(tf.int32)
        self.isTraining = tf.cast(self.isTraining, tf.bool)
        initializer = tf.contrib.layers.xavier_initializer()
        self.n_layer = 2
        self.weights={}
        for i in range(self.n_layer):
            self.weights['layer_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='JU_%d' % (i + 1))

        user_embeddings = self.user_embeddings
        item_embeddings = self.item_embeddings
        all_user_embeddings = [user_embeddings]
        all_item_embeddings = [item_embeddings]

        # message dropout.
        def without_dropout(embedding):
            return embedding

        def dropout(embedding):
            return tf.nn.dropout(embedding, rate=0.1)

        for i in range(self.n_layer):
            new_user_embeddings = tf.sparse_tensor_dense_matmul(H_u,self.user_embeddings)
            new_item_embeddings = tf.sparse_tensor_dense_matmul(H_i,self.item_embeddings)

            user_embeddings = tf.nn.leaky_relu(tf.matmul(new_user_embeddings,self.weights['layer_%d' %(i+1)])+ user_embeddings)
            item_embeddings = tf.nn.leaky_relu(tf.matmul(new_item_embeddings,self.weights['layer_%d' %(i+1)])+ item_embeddings)

            user_embeddings = tf.cond(self.isTraining, lambda: dropout(user_embeddings),
                                      lambda: without_dropout(user_embeddings))
            item_embeddings = tf.cond(self.isTraining, lambda: dropout(item_embeddings),
                                      lambda: without_dropout(item_embeddings))

            user_embeddings = tf.math.l2_normalize(user_embeddings,axis=1)
            item_embeddings = tf.math.l2_normalize(item_embeddings,axis=1)

            all_item_embeddings.append(item_embeddings)
            all_user_embeddings.append(user_embeddings)

        # user_embeddings = tf.reduce_sum(all_user_embeddings,axis=0)/(1+self.n_layer)
        # item_embeddings = tf.reduce_sum(all_item_embeddings, axis=0) / (1 + self.n_layer)
        #
        user_embeddings = tf.concat(all_user_embeddings,axis=1)
        item_embeddings = tf.concat(all_item_embeddings, axis=1)

        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_item_embedding = tf.nn.embedding_lookup(item_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(user_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(item_embeddings, self.v_idx)
        self.test = tf.reduce_sum(tf.multiply(self.u_embedding,item_embeddings),1)

    def trainModel(self):
        y = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1) \
            - tf.reduce_sum(tf.multiply(self.u_embedding, self.neg_item_embedding), 1)
        reg_loss = self.regU * (tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.v_embedding) +
                                                                    tf.nn.l2_loss(self.neg_item_embedding))
        for i in range(self.n_layer):
            reg_loss+= self.regU*tf.nn.l2_loss(self.weights['layer_%d' %(i+1)])
        loss = -tf.reduce_sum(tf.log(tf.sigmoid(y))) + reg_loss
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l = self.sess.run([train, loss],
                                feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx,self.isTraining:1})
                print('training:', epoch + 1, 'batch', n, 'loss:', l)

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.sess.run(self.test,feed_dict={self.u_idx:u,self.isTraining:0})
        else:
            return [self.data.globalMean] * self.num_items