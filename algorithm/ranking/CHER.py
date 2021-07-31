from baseclass.DeepRecommender import DeepRecommender
import tensorflow as tf
from math import sqrt
from utils import config
import os
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class CHER(DeepRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(CHER, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(CHER, self).readConfiguration()
        args = config.LineConfig(self.config['CHER'])
        self.ssl_reg = float(args['-lambda'])
        self.eps = float(args['-eps'])

    def _create_adv_inference(self):
        noises_u = tf.nn.l2_normalize(self.random_noises_U, 1) * self.eps
        noises_i = tf.nn.l2_normalize(self.random_noises_V, 1) * self.eps
        self.perturbed_user_embeddings = self.main_user_embeddings+noises_u#self.adv_U
        self.perturbed_item_embeddings = self.main_item_embeddings+noises_i#adv_V
        self.adv_ssl_loss = self.calc_ssl_loss()
        # # get gradients of Delta
        # self.grad_U, self.grad_V = tf.gradients(self.adv_ssl_loss, [self.adv_U, self.adv_V])
        #
        # # convert the IndexedSlice Data to Dense Tensor
        # self.grad_U_dense = tf.stop_gradient(self.grad_U)
        # self.grad_V_dense = tf.stop_gradient(self.grad_V)
        #
        # # normalization: new_grad = (grad / |grad|) * eps
        # self.update_U = self.adv_U.assign(tf.nn.l2_normalize(self.grad_U_dense, 1) * self.eps)
        # self.update_V = self.adv_V.assign(tf.nn.l2_normalize(self.grad_V_dense, 1) * self.eps)


    def initModel(self):
        super(CHER, self).initModel()
        initializer = tf.contrib.layers.xavier_initializer()
        self.user_embeddings = self.user_embeddings/2
        self.item_embeddings = self.item_embeddings/2
        self.ssl_rate = tf.placeholder(tf.float32)
        self.adv_U = tf.Variable(tf.zeros(shape=[self.num_users, self.embed_size]), dtype=tf.float32, trainable=False)
        self.adv_V = tf.Variable(tf.zeros(shape=[self.num_items, self.embed_size]), dtype=tf.float32, trainable=False)
        self.random_noises_U = tf.placeholder(tf.float32)
        self.random_noises_V = tf.placeholder(tf.float32)
        self.bi_matrix = tf.Variable(initializer([self.embed_size, self.embed_size]), name='bilinear')
        ego_embeddings = tf.concat([self.user_embeddings,self.item_embeddings], axis=0)
        self.n_layers = 2

        #main view
        indices = [[self.data.user[item[0]],self.num_users+self.data.item[item[1]]] for item in self.data.trainingData]
        indices += [[self.num_users+self.data.item[item[1]],self.data.user[item[0]]] for item in self.data.trainingData]
        values = [float(item[2])/sqrt(len(self.data.trainSet_u[item[0]]))/sqrt(len(self.data.trainSet_i[item[1]])) for item in self.data.trainingData]*2
        self.norm_adj = tf.SparseTensor(indices=indices, values=values, dense_shape=[self.num_users+self.num_items,self.num_users+self.num_items])

        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            ego_embeddings = tf.sparse_tensor_dense_matmul(self.norm_adj,ego_embeddings)
            all_embeddings += [ego_embeddings]
        #     # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]
        #
        all_embeddings = tf.reduce_sum(all_embeddings, axis=0)
        self.main_user_embeddings, self.main_item_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)

        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_item_embedding = tf.nn.embedding_lookup(self.main_item_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(self.main_user_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.main_item_embeddings, self.v_idx)
        self.test = tf.reduce_sum(tf.multiply(self.u_embedding,self.main_item_embeddings),1)


    def calc_ssl_loss(self):
        user_emb = tf.nn.embedding_lookup(self.main_user_embeddings, tf.unique(self.u_idx)[0])
        item_emb = tf.nn.embedding_lookup(self.main_item_embeddings, tf.unique(self.v_idx)[0])
        adv_user_emb = tf.nn.embedding_lookup(self.perturbed_user_embeddings, tf.unique(self.u_idx)[0])
        adv_item_emb = tf.nn.embedding_lookup(self.perturbed_item_embeddings, tf.unique(self.v_idx)[0])
        emb_merge1 = tf.concat([user_emb, item_emb], axis=0)
        emb_merge2 = tf.concat([adv_user_emb, adv_item_emb], axis=0)

        # cosine similarity
        normalize_emb_merge1 = tf.nn.l2_normalize(emb_merge1, 1)
        normalize_emb_merge2 = tf.nn.l2_normalize(emb_merge2, 1)

        pos_score = tf.reduce_sum(tf.multiply(normalize_emb_merge1, normalize_emb_merge2), axis=1)
        ttl_score = tf.matmul(normalize_emb_merge1, normalize_emb_merge2, transpose_a=False, transpose_b=True)

        pos_score = tf.exp(pos_score /0.1)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score /0.1), axis=1)
        ssl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))
        ssl_loss = ssl_loss
        return self.ssl_rate*ssl_loss


    def saveModel(self):
        # store the best parameters
        self.bestU, self.bestV = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])

    def buildModel(self):
        self._create_adv_inference()
        #main task: recommendation
        y = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1) \
            - tf.reduce_sum(tf.multiply(self.u_embedding, self.neg_item_embedding), 1)
        rec_loss = -tf.reduce_sum(tf.log(tf.sigmoid(y))) + self.regU * (tf.nn.l2_loss(self.user_embeddings) +
                                                                    tf.nn.l2_loss(self.item_embeddings))

        #SSL task: contrastive learning
        loss = rec_loss+ self.adv_ssl_loss
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        for iteration in range(self.maxIter):
            for n, batch in enumerate(self.next_batch_pairwise()):
                noise_u = np.random.random((self.num_users, self.embed_size))
                noise_i = np.random.random((self.num_items, self.embed_size))
                # if iteration < self.maxIter/3:
                #     user_idx, i_idx, j_idx = batch
                #     #self.sess.run([self.update_U, self.update_V], feed_dict={self.u_idx: user_idx, self.v_idx: i_idx})
                #     _, l, rec_l, ssl_l = self.sess.run([train, loss, rec_loss, self.adv_ssl_loss],
                #                                        feed_dict={self.random_noises_U:noise_u,self.random_noises_V:noise_i,self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx,self.ssl_rate:0})
                #
                # else:
                user_idx, i_idx, j_idx = batch
                #self.sess.run([self.update_U, self.update_V],feed_dict={self.u_idx: user_idx,  self.v_idx: i_idx})
                _, l,rec_l,ssl_l = self.sess.run([train, loss, rec_loss, self.adv_ssl_loss],
                                feed_dict={self.random_noises_U:noise_u,self.random_noises_V:noise_i,self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx,self.ssl_rate:self.ssl_reg})
                print('training:', iteration + 1, 'batch', n, 'total_loss:',l, 'rec_loss:', rec_l,'ssl_loss',ssl_l)
            if iteration > 0:
                self.U, self.V = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])
                self.ranking_performance(iteration)
        self.U, self.V = self.bestU, self.bestV

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items