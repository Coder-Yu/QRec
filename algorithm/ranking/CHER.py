from baseclass.DeepRecommender import DeepRecommender
import tensorflow as tf
from math import sqrt
from scipy.sparse import coo_matrix,csr_matrix
from tool import config
import numpy as np
import random
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class CHER(DeepRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(CHER, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(CHER, self).readConfiguration()
        args = config.LineConfig(self.config['CHER'])
        self.reg_lambda = float(args['-lambda'])
        self.eps = float(args['-eps'])

    def _create_adv_inference(self):
        self.perturbed_user_embeddings = self.main_user_embeddings+self.adv_U
        self.perturbed_item_embeddings = self.main_item_embeddings+self.adv_V

        # ssl view
        # perturbed_ego_embeddings = tf.concat([self.perturbed_user_embeddings,self.perturbed_item_embeddings], axis=0)
        # all_embeddings = [perturbed_ego_embeddings]
        # for k in range(self.n_layers):
        #     perturbed_ego_embeddings = tf.sparse_tensor_dense_matmul(self.norm_adj, perturbed_ego_embeddings)
        #     # normalize the distribution of embeddings.
        #     norm_embeddings = tf.math.l2_normalize(perturbed_ego_embeddings, axis=1)
        #     all_embeddings += [norm_embeddings]
        # all_embeddings = tf.reduce_sum(all_embeddings, axis=0)
        #
        # self.ssl_user_embeddings, self.ssl_item_embeddings = tf.split(all_embeddings,
        #                                                                 [self.num_users, self.num_items], 0)

        self.adv_ssl_loss = self.mutual_information_maximization(tf.nn.embedding_lookup(self.main_user_embeddings,tf.unique(self.u_idx)[0]),
                                                                 tf.nn.embedding_lookup(self.perturbed_user_embeddings,tf.unique(self.u_idx)[0]))
        self.adv_ssl_loss += self.mutual_information_maximization(tf.nn.embedding_lookup(self.main_item_embeddings,tf.unique(self.v_idx)[0]),
                                                                 tf.nn.embedding_lookup(self.perturbed_item_embeddings,tf.unique(self.v_idx)[0]))
        # get gradients of Delta
        self.grad_U, self.grad_V = tf.gradients(self.adv_ssl_loss, [self.adv_U, self.adv_V])

        # convert the IndexedSlice Data to Dense Tensor
        self.grad_U_dense = tf.stop_gradient(self.grad_U)
        self.grad_V_dense = tf.stop_gradient(self.grad_V)

        # normalization: new_grad = (grad / |grad|) * eps
        self.update_U = self.adv_U.assign(tf.nn.l2_normalize(self.grad_U_dense, 1) * self.eps)
        self.update_V = self.adv_V.assign(tf.nn.l2_normalize(self.grad_V_dense, 1) * self.eps)


    def initModel(self):
        super(CHER, self).initModel()
        initializer = tf.contrib.layers.xavier_initializer()
        self.ssl_rate = tf.placeholder(tf.float32)
        self.adv_U = tf.Variable(tf.zeros(shape=[self.num_users, self.embed_size]), dtype=tf.float32, trainable=False)
        self.adv_V = tf.Variable(tf.zeros(shape=[self.num_items, self.embed_size]), dtype=tf.float32, trainable=False)
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
            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]
        all_embeddings = tf.reduce_sum(all_embeddings, axis=0)

        self.main_user_embeddings, self.main_item_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)

        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_item_embedding = tf.nn.embedding_lookup(self.main_item_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(self.main_user_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.main_item_embeddings, self.v_idx)
        self.test = tf.reduce_sum(tf.multiply(self.u_embedding,self.main_item_embeddings),1)



    def mutual_information_maximization(self,em1,em2):
        def row_shuffle(embedding):
            return tf.gather(embedding, tf.random.shuffle(tf.range(tf.shape(embedding)[0])))
        def row_column_shuffle(embedding):
            corrupted_embedding = tf.transpose(tf.gather(tf.transpose(embedding), tf.random.shuffle(tf.range(tf.shape(tf.transpose(embedding))[0]))))
            corrupted_embedding = tf.gather(corrupted_embedding, tf.random.shuffle(tf.range(tf.shape(corrupted_embedding)[0])))
            return corrupted_embedding
        def cosine(x1,x2):
            #x1 = tf.nn.l2_normalize(x1, 1)
            #x2 = tf.nn.l2_normalize(x2, 1)
            x1 = tf.matmul(x1,self.bi_matrix)
            x2 = tf.matmul(x2,self.bi_matrix)
            return tf.reduce_sum(tf.multiply(x1,x2),1)
        pos = cosine(em1,em2)
        neg = cosine(em1,row_column_shuffle(em2))
        loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos/0.2)+1e-6)-tf.log(1-tf.sigmoid(neg/0.2)+1e-6))
        return loss

    def saveModel(self):
        # store the best parameters
        self.bestU, self.bestV = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])

    def buildModel(self):
        self._create_adv_inference()
        #main task: recommendation
        y = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1) \
            - tf.reduce_sum(tf.multiply(self.u_embedding, self.neg_item_embedding), 1)
        rec_loss = -tf.reduce_sum(tf.log(tf.sigmoid(y))) + self.regU * (tf.nn.l2_loss(self.u_embedding) +
                                                                    tf.nn.l2_loss(self.v_embedding) +
                                                                    tf.nn.l2_loss(self.neg_item_embedding))
        #SSL task: contrastive learning
        loss = rec_loss+ self.ssl_rate*self.adv_ssl_loss
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        for iteration in range(self.maxIter):
            for n, batch in enumerate(self.next_batch_pairwise()):
                if iteration <= self.maxIter-50:
                    user_idx, i_idx, j_idx = batch
                    _, l, rec_l, ssl_l = self.sess.run([train, loss, rec_loss, self.adv_ssl_loss],
                                                       feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx,
                                                                  self.v_idx: i_idx, self.ssl_rate:0})
                else:
                    user_idx, i_idx, j_idx = batch
                    self.sess.run([self.update_U, self.update_V],feed_dict={self.u_idx: user_idx,  self.v_idx: i_idx})
                    _, l,rec_l,ssl_l = self.sess.run([train, loss, rec_loss, self.adv_ssl_loss],
                                    feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx,self.ssl_rate:self.reg_lambda})
                print 'training:', iteration + 1, 'batch', n, 'total_loss:',l, 'rec_loss:', rec_l,'ssl_loss',self.reg_lambda*ssl_l
            if iteration > self.maxIter - 50:
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