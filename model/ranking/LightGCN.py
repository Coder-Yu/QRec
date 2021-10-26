from base.graphRecommender import GraphRecommender
import tensorflow as tf
from math import sqrt
from util.config import LineConfig
class LightGCN(GraphRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(LightGCN, self).__init__(conf,trainingSet,testSet,fold)
        args = LineConfig(self.config['LightGCN'])
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

    def buildModel(self):
        y = tf.reduce_sum(tf.multiply(self.batch_user_emb, self.batch_pos_item_emb), 1) \
            - tf.reduce_sum(tf.multiply(self.batch_user_emb, self.batch_neg_item_emb), 1)
        loss = -tf.reduce_sum(tf.log(tf.sigmoid(y)+10e-6)) + self.regU * (tf.nn.l2_loss(self.batch_user_emb) + tf.nn.l2_loss(self.batch_pos_item_emb)+ tf.nn.l2_loss(self.batch_neg_item_emb))
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        import time
        for epoch in range(self.maxEpoch):
            s = time.time()
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l = self.sess.run([train, loss],
                                feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print(self.foldInfo,'training:', epoch + 1, 'batch', n, 'loss:', l)
            self.U, self.V = self.sess.run([self.multi_user_embeddings, self.multi_item_embeddings])
            e = time.time()
            print("Epoch run time: %f s" % (e - s))

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items