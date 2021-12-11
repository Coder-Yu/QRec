from base.graphRecommender import GraphRecommender
import tensorflow as tf
from util.loss import bpr_loss
class NGCF(GraphRecommender):

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(NGCF, self).__init__(conf,trainingSet,testSet,fold)

    def initModel(self):
        super(NGCF, self).initModel()
        self.isTraining = tf.placeholder(tf.int32)
        self.isTraining = tf.cast(self.isTraining, tf.bool)
        ego_embeddings = tf.concat([self.user_embeddings,self.item_embeddings], axis=0)
        norm_adj = self.create_joint_sparse_adj_tensor()
        self.weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        weight_size = [self.emb_size, self.emb_size, self.emb_size] #can be changed
        weight_size_list = [self.emb_size] + weight_size
        self.n_layers = 2
        #initialize parameters
        for k in range(self.n_layers):
            self.weights['W_%d_1' % k] = tf.Variable(
                initializer([weight_size_list[k], weight_size_list[k + 1]]), name='W_%d_1' % k)
            self.weights['W_%d_2' % k] = tf.Variable(
                initializer([weight_size_list[k], weight_size_list[k + 1]]), name='W_%d_2' % k)
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            side_embeddings = tf.sparse_tensor_dense_matmul(norm_adj,ego_embeddings)
            sum_embeddings = tf.matmul(side_embeddings+ego_embeddings, self.weights['W_%d_1' % k])
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            bi_embeddings = tf.matmul(bi_embeddings, self.weights['W_%d_2' % k])
            ego_embeddings = tf.nn.leaky_relu(sum_embeddings+bi_embeddings)
            # message dropout.
            def without_dropout():
                return ego_embeddings
            def dropout():
                return tf.nn.dropout(ego_embeddings, keep_prob=0.9)
            ego_embeddings = tf.cond(self.isTraining,lambda:dropout(),lambda:without_dropout())
            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        self.multi_user_embeddings, self.multi_item_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.batch_neg_item_emb = tf.nn.embedding_lookup(self.multi_item_embeddings, self.neg_idx)
        self.batch_user_emb = tf.nn.embedding_lookup(self.multi_user_embeddings, self.u_idx)
        self.batch_pos_item_emb = tf.nn.embedding_lookup(self.multi_item_embeddings, self.v_idx)
        self.test = tf.reduce_sum(tf.multiply(self.batch_user_emb,self.multi_item_embeddings),1)

    def trainModel(self):
        rec_loss = bpr_loss(self.batch_user_emb, self.batch_pos_item_emb, self.batch_neg_item_emb)
        rec_loss += self.regU * (tf.nn.l2_loss(self.batch_user_emb) + tf.nn.l2_loss(self.batch_pos_item_emb) + tf.nn.l2_loss(
                self.batch_neg_item_emb))
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(rec_loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l = self.sess.run([train, rec_loss],
                                feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx,self.isTraining:1})
                print('training:', epoch + 1, 'batch', n, 'loss:', l)

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.sess.run(self.test,feed_dict={self.u_idx:u,self.isTraining:0})
        else:
            return [self.data.globalMean] * self.num_items