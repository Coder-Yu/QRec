from base.deepRecommender import DeepRecommender
import tensorflow as tf
from math import sqrt
from util.config import LineConfig
class LightGCN(DeepRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(LightGCN, self).__init__(conf,trainingSet,testSet,fold)
        args = LineConfig(self.config['LightGCN'])
        self.n_layers = int(args['-n_layer'])

    def initModel(self):
        super(LightGCN, self).initModel()
        initializer = tf.contrib.layers.xavier_initializer()
        # self.user_embeddings = tf.Variable(initializer(shape=[self.num_users, self.emb_size]), name='U')
        # self.item_embeddings = tf.Variable(initializer(shape=[self.num_items, self.emb_size]), name='V')
        ego_embeddings = tf.concat([self.user_embeddings,self.item_embeddings], axis=0)
        indices = [[self.data.user[item[0]],self.num_users+self.data.item[item[1]]] for item in self.data.trainingData]
        indices += [[self.num_users+self.data.item[item[1]],self.data.user[item[0]]] for item in self.data.trainingData]
        values = [float(item[2])/sqrt(len(self.data.trainSet_u[item[0]]))/sqrt(len(self.data.trainSet_i[item[1]])) for item in self.data.trainingData]*2
        norm_adj = tf.SparseTensor(indices=indices, values=values, dense_shape=[self.num_users+self.num_items,self.num_users+self.num_items])

        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            ego_embeddings = tf.sparse_tensor_dense_matmul(norm_adj,ego_embeddings)
            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
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
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l = self.sess.run([train, loss],
                                feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print(self.foldInfo,'training:', epoch + 1, 'batch', n, 'loss:', l)
            self.U, self.V = self.sess.run([self.multi_user_embeddings, self.multi_item_embeddings])
            self.ranking_performance(epoch)
        # self.U, self.V = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])

        self.U, self.V = self.bestU, self.bestV
        import pickle
        ue = open('user_lgcn.emb','wb')
        pickle.dump(self.U,ue)
        ie = open('item_lgcn.emb','wb')
        pickle.dump(self.V,ie)

    def saveModel(self):
        self.bestU, self.bestV = self.sess.run([self.multi_user_embeddings, self.multi_item_embeddings])
    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items