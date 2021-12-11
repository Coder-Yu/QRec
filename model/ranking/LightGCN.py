from base.graphRecommender import GraphRecommender
import tensorflow as tf
from util.loss import bpr_loss
from util.config import OptionConf
class LightGCN(GraphRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(LightGCN, self).__init__(conf,trainingSet,testSet,fold)
        args = OptionConf(self.config['LightGCN'])
        self.n_layers = int(args['-n_layer'])

    def initModel(self):
        super(LightGCN, self).initModel()
        ego_embeddings = tf.concat([self.user_embeddings,self.item_embeddings], axis=0)
        norm_adj = self.create_joint_sparse_adj_tensor()
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            ego_embeddings = tf.sparse_tensor_dense_matmul(norm_adj,ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = tf.reduce_mean(all_embeddings, axis=0)
        self.multi_user_embeddings, self.multi_item_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.batch_neg_item_emb = tf.nn.embedding_lookup(self.multi_item_embeddings, self.neg_idx)
        self.batch_user_emb = tf.nn.embedding_lookup(self.multi_user_embeddings, self.u_idx)
        self.batch_pos_item_emb = tf.nn.embedding_lookup(self.multi_item_embeddings, self.v_idx)
        self.test = tf.reduce_sum(tf.multiply(self.batch_user_emb, self.multi_item_embeddings), 1)

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
                                feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print(self.foldInfo,'training:', epoch + 1, 'batch', n, 'loss:', l)
        self.U, self.V = self.sess.run([self.multi_user_embeddings, self.multi_item_embeddings])

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items