from base.deepRecommender import DeepRecommender
import tensorflow as tf
from math import sqrt
from util import config
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
        self.n_layers = int(args['-n_layer'])

    # def _create_perburbed_inference(self):
    #     #batch_user_emb = tf.nn.embedding_lookup(self.main_user_embeddings, tf.unique(self.u_idx)[0])
    #     #shape_u = tf.shape(self.user_embeddings)
    #     #batch_item_emb = tf.nn.embedding_lookup(self.main_item_embeddings, tf.unique(self.v_idx)[0])
    #     # shape_i = tf.shape(self.item_embeddings)
    #     # noise_u = tf.random.uniform(shape=shape_u)
    #     # noise_i = tf.random.uniform(shape=shape_i)
    #     # noises_u = tf.multiply(tf.math.sign(self.user_embeddings),tf.nn.l2_normalize(noise_u, 1) * self.eps)
    #     # noises_i = tf.multiply(tf.math.sign(self.item_embeddings),tf.nn.l2_normalize(noise_i, 1) * self.eps)
    #     # self.perturbed_user_emb = self.user_embeddings+noises_u#self.perburbed_U
    #     # self.perturbed_item_emb = self.item_embeddings+noises_i#perburbed_V
    #
    #     # # get gradients of Delta
    #     self.grad_U, self.grad_V = tf.gradients(self.calc_adv_ssl_loss(), [self.adv_U, self.adv_V])
    #     #
    #     # # convert the IndexedSlice Data to Dense Tensor
    #     self.grad_U_dense = tf.stop_gradient(self.grad_U)
    #     self.grad_V_dense = tf.stop_gradient(self.grad_V)
    #     #
    #     # # normalization: new_grad = (grad / |grad|) * eps
    #     self.update_U = self.adv_U.assign(tf.math.sign(self.grad_U_dense) * self.eps)
    #     self.update_V = self.adv_V.assign(tf.math.sign(self.grad_V_dense) * self.eps)


    def initModel(self):
        super(CHER, self).initModel()
        # self.user_embeddings = self.user_embeddings/2
        # self.item_embeddings = self.item_embeddings/2
        self.ssl_rate = tf.placeholder(tf.float32)
        self.adv_U = tf.Variable(tf.zeros(shape=[self.num_users, self.emb_size]), dtype=tf.float32)
        self.adv_V = tf.Variable(tf.zeros(shape=[self.num_items, self.emb_size]), dtype=tf.float32)
        # self.random_noises_U = tf.placeholder(tf.float32)
        # self.random_noises_V = tf.placeholder(tf.float32)
        #self._create_perburbed_inference()
        ego_embeddings = tf.concat([self.user_embeddings,self.item_embeddings], axis=0)
        #p_ego_embeddings = tf.concat([self.user_embeddings+self.adv_U,self.item_embeddings+self.adv_V], axis=0)
        random_noise = tf.random.normal(ego_embeddings.shape)
        n_ego_embeddings = tf.concat([self.user_embeddings,self.item_embeddings], axis=0)
        n_ego_embeddings += tf.multiply(tf.math.sign(n_ego_embeddings),tf.nn.l2_normalize(random_noise, 1) * self.eps)
        random_noise = tf.random.uniform(ego_embeddings.shape)
        p_ego_embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        p_ego_embeddings += tf.multiply(tf.math.sign(n_ego_embeddings), tf.nn.l2_normalize(random_noise, 1) * self.eps)
        #main view
        indices = [[self.data.user[item[0]],self.num_users+self.data.item[item[1]]] for item in self.data.trainingData]
        indices += [[self.num_users+self.data.item[item[1]],self.data.user[item[0]]] for item in self.data.trainingData]
        values = [float(item[2])/sqrt(len(self.data.trainSet_u[item[0]]))/sqrt(len(self.data.trainSet_i[item[1]])) for item in self.data.trainingData]*2
        self.norm_adj = tf.SparseTensor(indices=indices, values=values, dense_shape=[self.num_users+self.num_items,self.num_users+self.num_items])

        all_embeddings = [ego_embeddings]
        p_all_embeddings = [p_ego_embeddings]
        n_all_embeddings = [n_ego_embeddings]
        for k in range(self.n_layers):
            ego_embeddings = tf.sparse_tensor_dense_matmul(self.norm_adj,ego_embeddings)
            n_ego_embeddings = tf.sparse_tensor_dense_matmul(self.norm_adj,n_ego_embeddings)
            random_noise = tf.random.normal(n_ego_embeddings.shape)
            n_ego_embeddings +=  tf.multiply(tf.math.sign(n_ego_embeddings),tf.nn.l2_normalize(random_noise, 1) * self.eps)
            p_ego_embeddings = tf.sparse_tensor_dense_matmul(self.norm_adj,p_ego_embeddings)
            #all_embeddings += [ego_embeddings]
        #     # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [ego_embeddings] #converge faster

            norm_embeddings = tf.math.l2_normalize(n_ego_embeddings, axis=1)
            n_all_embeddings += [n_ego_embeddings] #converge faster
            #
            # norm_embeddings = tf.math.l2_normalize(n_ego_embeddings, axis=1)
            p_all_embeddings += [p_ego_embeddings] #converge faster
        #
        all_embeddings = tf.reduce_mean(all_embeddings, axis=0)
        self.main_user_embeddings, self.main_item_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
        n_all_embeddings = tf.reduce_mean(n_all_embeddings, axis=0)
        self.noised_user_embeddings, self.noised_item_embeddings = tf.split(n_all_embeddings,[self.num_users, self.num_items], 0)
        p_all_embeddings = tf.reduce_mean(p_all_embeddings, axis=0)
        self.perturbed_user_embeddings, self.perturbed_item_embeddings = tf.split(p_all_embeddings,[self.num_users, self.num_items], 0)

        # noise_u = tf.random.normal(self.adv_U.shape)
        # noise_i = tf.random.normal(self.adv_V.shape)
        # noises_u = tf.multiply(tf.math.sign(self.main_user_embeddings),
        #                        tf.nn.l2_normalize(noise_u, 1) * self.eps)
        # noises_i = tf.multiply(tf.math.sign(self.main_item_embeddings),
        #                        tf.nn.l2_normalize(noise_i, 1) * self.eps)
        # self.noised_user_embeddings = self.main_user_embeddings + noises_u  # self.perburbed_U
        # self.noised_item_embeddings = self.main_item_embeddings + noises_i  # perburbed_V
        # self.perturbed_user_embeddings = self.noised_user_embeddings+self.adv_U
        # self.perturbed_item_embeddings = self.noised_item_embeddings+self.adv_V
        #self._create_perburbed_inference()
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_item_embedding = tf.nn.embedding_lookup(self.main_item_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(self.main_user_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.main_item_embeddings, self.v_idx)

    def saveModel(self):
        self.bestU, self.bestV = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])

    def calc_ssl_loss(self):
        # user_emb = tf.nn.embedding_lookup(self.main_user_embeddings, tf.unique(self.u_idx)[0])
        # item_emb = tf.nn.embedding_lookup(self.main_item_embeddings, tf.unique(self.v_idx)[0])
        perturbed_user_emb = tf.nn.embedding_lookup(self.perturbed_user_embeddings, tf.unique(self.u_idx)[0])
        perturbed_item_emb = tf.nn.embedding_lookup(self.perturbed_item_embeddings, tf.unique(self.v_idx)[0])
        user_emb = tf.nn.embedding_lookup(self.noised_user_embeddings, tf.unique(self.u_idx)[0])
        item_emb = tf.nn.embedding_lookup(self.noised_item_embeddings, tf.unique(self.v_idx)[0])
        # emb_merge1 = tf.concat([user_emb, item_emb], axis=0)
        # emb_merge2 = tf.concat([perturbed_user_emb, perturbed_item_emb], axis=0)
        #
        # # cosine similarity
        # normalize_emb_merge1 = tf.nn.l2_normalize(emb_merge1, 1)
        # normalize_emb_merge2 = tf.nn.l2_normalize(emb_merge2, 1)
        normalize_emb_user = tf.nn.l2_normalize(user_emb, 1)
        normalize_emb_user_p = tf.nn.l2_normalize(perturbed_user_emb, 1)
        normalize_emb_item = tf.nn.l2_normalize(item_emb, 1)
        normalize_emb_item_p = tf.nn.l2_normalize(perturbed_item_emb, 1)

        pos_score_u = tf.reduce_sum(tf.multiply(normalize_emb_user, normalize_emb_user_p), axis=1)
        pos_score_i = tf.reduce_sum(tf.multiply(normalize_emb_item, normalize_emb_item_p), axis=1)
        # self.pos_score = pos_score
        ttl_score_u = tf.matmul(normalize_emb_user, normalize_emb_user_p, transpose_a=False, transpose_b=True)
        ttl_score_i = tf.matmul(normalize_emb_item, normalize_emb_item_p, transpose_a=False, transpose_b=True)
        #ttl_score = tf.concat([ttl_score,tf.reshape(pos_score,(tf.shape(pos_score)[0],1))],1)
        pos_score_u = tf.exp(pos_score_u /0.2)
        ttl_score_u = tf.reduce_sum(tf.exp(ttl_score_u /0.2), axis=1)
        pos_score_i = tf.exp(pos_score_i /0.2)
        ttl_score_i = tf.reduce_sum(tf.exp(ttl_score_i /0.2), axis=1)
        ssl_loss = -tf.reduce_sum(tf.log(pos_score_u / ttl_score_u)) -tf.reduce_sum(tf.log(pos_score_i / ttl_score_i))
        return self.ssl_rate*ssl_loss

    # def calc_ssl_loss(self):
    #     n_user_emb = tf.nn.embedding_lookup(self.perturbed_user_embeddings, tf.unique(self.u_idx)[0])
    #     n_item_emb = tf.nn.embedding_lookup(self.perturbed_item_embeddings, tf.unique(self.v_idx)[0])
    #     user_emb = tf.nn.embedding_lookup(self.user_embeddings, tf.unique(self.u_idx)[0])
    #     item_emb = tf.nn.embedding_lookup(self.item_embeddings, tf.unique(self.v_idx)[0])
    #     emb_merge1 = tf.concat([user_emb, item_emb], axis=0)
    #     emb_merge2 = tf.concat([n_user_emb, n_item_emb], axis=0)
    #
    #     # cosine similarity
    #     normalize_emb_merge1 = tf.nn.l2_normalize(emb_merge1, 1)
    #     normalize_emb_merge2 = tf.nn.l2_normalize(emb_merge2, 1)
    #
    #     pos_score = tf.reduce_sum(tf.multiply(normalize_emb_merge1, normalize_emb_merge2), axis=1)
    #     ttl_score = tf.matmul(normalize_emb_merge1, normalize_emb_merge2, transpose_a=False, transpose_b=True)
    #     #ttl_score = tf.concat([ttl_score,tf.reshape(pos_score,(tf.shape(pos_score)[0],1))],1)
    #     diag = tf.diag_part(ttl_score)
    #     ttl_score = tf.matrix_diag(-diag)+ttl_score
    #     pos_score = tf.exp(pos_score /0.2)
    #     ttl_score = tf.reduce_sum(tf.exp(ttl_score /0.2), axis=1)
    #     ssl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))
    #     return self.ssl_rate*ssl_loss

    def buildModel(self):
        #self._create_perburbed_inference()
        #self.ssl_loss = self.calc_ssl_loss()
        self.perturbed_ssl_loss = self.calc_ssl_loss()
        #main task: recommendation
        y = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1) \
            - tf.reduce_sum(tf.multiply(self.u_embedding, self.neg_item_embedding), 1)
        rec_loss = -tf.reduce_sum(tf.log(tf.sigmoid(y))) + self.regU * (tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.item_embeddings))
        #SSL task: contrastive learning
        #con_loss = self._create_uniform_noise()
        loss = rec_loss + self.perturbed_ssl_loss
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                # noise_u = np.random.random((self.num_users, self.emb_size))
                # noise_i = np.random.random((self.num_items, self.emb_size))
                # if epoch < self.maxIter/3:
                #     user_idx, i_idx, j_idx = batch
                #     #self.sess.run([self.update_U, self.update_V], feed_dict={self.u_idx: user_idx, self.v_idx: i_idx})
                #     _, l, rec_l, ssl_l = self.sess.run([train, loss, rec_loss, self.perburbed_ssl_loss],
                #                                        feed_dict={self.random_noises_U:noise_u,self.random_noises_V:noise_i,self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx,self.ssl_rate:0})
                #
                # else:
                user_idx, i_idx, j_idx = batch
                #self.sess.run([self.update_U, self.update_V],feed_dict={self.u_idx: user_idx,  self.v_idx: i_idx})
                # _, l,rec_l,ssl_l = self.sess.run([train, loss, rec_loss, self.perburbed_ssl_loss],
                #                 feed_dict={self.random_noises_U:noise_u,self.random_noises_V:noise_i,self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx,self.ssl_rate:self.ssl_reg})
                _, l, rec_l, ssl_l = self.sess.run([train, loss, rec_loss, self.perturbed_ssl_loss],
                                                   feed_dict={self.u_idx: user_idx,
                                                              self.neg_idx: j_idx, self.v_idx: i_idx,
                                                              self.ssl_rate: self.ssl_reg})
                # pos = self.sess.run(self.pos_score,
                #                                    feed_dict={self.u_idx: user_idx,
                #                                               self.neg_idx: j_idx, self.v_idx: i_idx,
                #                                               self.ssl_rate: self.ssl_reg})
                #print (pos)
                print('training:', epoch + 1, 'batch', n, 'total_loss:',l, 'rec_loss:', rec_l,'ssl_loss',ssl_l)
            self.U, self.V = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])
            self.ranking_performance(epoch)
        #self.U, self.V = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])
        self.U,self.V = self.bestU,self.bestV

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items