from base.graphRecommender import GraphRecommender
import tensorflow as tf
from util import config
from util.loss import bpr_loss
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class GACL(GraphRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(GACL, self).__init__(conf, trainingSet, testSet, fold)
        self.performance = []
    def readConfiguration(self):
        super(GACL, self).readConfiguration()
        args = config.OptionConf(self.config['GACL'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])

    def initModel(self):
        super(GACL, self).initModel()
        initializer = tf.contrib.layers.xavier_initializer()
        self.user_embeddings = tf.Variable(initializer([self.num_users, self.emb_size]))
        self.item_embeddings = tf.Variable(initializer([self.num_items, self.emb_size]))
        ego_embeddings = tf.concat([self.user_embeddings,self.item_embeddings], axis=0)
        random_noise = tf.random.uniform(ego_embeddings.shape)
        n_ego_embeddings1 = ego_embeddings + tf.multiply(tf.sign(ego_embeddings),tf.nn.l2_normalize(random_noise, 1)) * self.eps
        random_noise = tf.random.uniform(ego_embeddings.shape)
        n_ego_embeddings2 = ego_embeddings + tf.multiply(tf.sign(ego_embeddings),tf.nn.l2_normalize(random_noise, 1)) * self.eps
        #adjaceny matrix
        self.norm_adj = self.create_joint_sparse_adj_tensor()
        all_embeddings = [ego_embeddings]
        n_all_embeddings1 = [n_ego_embeddings1]
        n_all_embeddings2 = [n_ego_embeddings2]
        for k in range(self.n_layers):
            ego_embeddings = tf.sparse_tensor_dense_matmul(self.norm_adj,ego_embeddings)
            n_ego_embeddings1 = tf.sparse_tensor_dense_matmul(self.norm_adj,n_ego_embeddings1)
            n_ego_embeddings2 = tf.sparse_tensor_dense_matmul(self.norm_adj,n_ego_embeddings2)
            random_noise = tf.random.uniform(n_ego_embeddings1.shape)
            n_ego_embeddings1 += tf.multiply(tf.sign(n_ego_embeddings1),tf.nn.l2_normalize(random_noise, 1)) * self.eps
            random_noise = tf.random.uniform(n_ego_embeddings2.shape)
            n_ego_embeddings2 += tf.multiply(tf.sign(n_ego_embeddings2),tf.nn.l2_normalize(random_noise, 1)) * self.eps
            all_embeddings += [ego_embeddings]
            n_all_embeddings1 += [n_ego_embeddings1]
            n_all_embeddings2 += [n_ego_embeddings2]
        all_embeddings = tf.reduce_mean(all_embeddings, axis=0)
        self.main_user_embeddings, self.main_item_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
        n_all_embeddings1 = tf.reduce_mean(n_all_embeddings1, axis=0)
        self.perturbed_user_embeddings1, self.perturbed_item_embeddings1 = tf.split(n_all_embeddings1,[self.num_users, self.num_items], 0)
        n_all_embeddings2 = tf.reduce_mean(n_all_embeddings2, axis=0)
        self.perturbed_user_embeddings2, self.perturbed_item_embeddings2  = tf.split(n_all_embeddings2,[self.num_users, self.num_items], 0)

        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.batch_neg_item_emb = tf.nn.embedding_lookup(self.main_item_embeddings, self.neg_idx)
        self.batch_user_emb = tf.nn.embedding_lookup(self.main_user_embeddings, self.u_idx)
        self.batch_pos_item_emb = tf.nn.embedding_lookup(self.main_item_embeddings, self.v_idx)
        #self.uniformity = self.uniformity_measure(tf.nn.embedding_lookup(self.main_item_embeddings, tf.unique(self.v_idx)[0]))

    # def uniformity_measure(self,emb):
    #     #measure the uniformity of embeddings in a batch
    #     emb = tf.nn.l2_normalize(emb, 1)
    #     return tf.log(tf.reduce_sum(tf.exp(tf.matmul(emb,emb,transpose_b=True)-1))/tf.cast(tf.shape(emb)[0]**2,tf.float32))


    def saveModel(self):
        self.bestU, self.bestV = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])

    def calc_cl_loss(self):
        p_user_emb1 = tf.nn.embedding_lookup(self.perturbed_user_embeddings1, tf.unique(self.u_idx)[0])
        p_item_emb1 = tf.nn.embedding_lookup(self.perturbed_item_embeddings1, tf.unique(self.v_idx)[0])
        p_user_emb2 = tf.nn.embedding_lookup(self.perturbed_user_embeddings2, tf.unique(self.u_idx)[0])
        p_item_emb2 = tf.nn.embedding_lookup(self.perturbed_item_embeddings2, tf.unique(self.v_idx)[0])

        #hybrid contrast
        # emb_merge1 = tf.concat([p_user_emb1, p_item_emb1], axis=0)
        # emb_merge2 = tf.concat([p_user_emb2, p_item_emb2], axis=0)
        # normalize_emb_merge1 = tf.nn.l2_normalize(emb_merge1, 1)
        # normalize_emb_merge2 = tf.nn.l2_normalize(emb_merge2, 1)
        # pos_score = tf.reduce_sum(tf.multiply(normalize_emb_merge1, normalize_emb_merge2), axis=1)
        # ttl_score = tf.matmul(normalize_emb_merge1, normalize_emb_merge2, transpose_a=False, transpose_b=True)
        # pos_score = tf.exp(pos_score /0.2)
        # ttl_score = tf.reduce_sum(tf.exp(ttl_score /0.2), axis=1)
        # cl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))

        #group contrast
        normalize_emb_user1 = tf.nn.l2_normalize(p_user_emb1, 1)
        normalize_emb_user2 = tf.nn.l2_normalize(p_user_emb2, 1)
        normalize_emb_item1 = tf.nn.l2_normalize(p_item_emb1, 1)
        normalize_emb_item2 = tf.nn.l2_normalize(p_item_emb2, 1)
        pos_score_u = tf.reduce_sum(tf.multiply(normalize_emb_user1, normalize_emb_user2), axis=1)
        pos_score_i = tf.reduce_sum(tf.multiply(normalize_emb_item1, normalize_emb_item2), axis=1)
        ttl_score_u = tf.matmul(normalize_emb_user1, normalize_emb_user2, transpose_a=False, transpose_b=True)
        ttl_score_i = tf.matmul(normalize_emb_item1, normalize_emb_item2, transpose_a=False, transpose_b=True)
        pos_score_u = tf.exp(pos_score_u /0.2)
        ttl_score_u = tf.reduce_sum(tf.exp(ttl_score_u /0.2), axis=1)
        pos_score_i = tf.exp(pos_score_i /0.2)
        ttl_score_i = tf.reduce_sum(tf.exp(ttl_score_i /0.2), axis=1)
        cl_loss = -tf.reduce_sum(tf.log(pos_score_u/ttl_score_u))-tf.reduce_sum(tf.log(pos_score_i/ttl_score_i))
        return self.cl_rate*cl_loss

    def buildModel(self):        
        #main task: recommendation
        rec_loss = bpr_loss(self.batch_user_emb,self.batch_pos_item_emb,self.batch_neg_item_emb)
        rec_loss += self.regU * (tf.nn.l2_loss(self.batch_user_emb) + tf.nn.l2_loss(self.batch_pos_item_emb) + tf.nn.l2_loss(self.batch_neg_item_emb))
        #CL task
        self.cl_loss = self.calc_cl_loss()
        loss = rec_loss+self.cl_loss
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l, rec_l, cl_l = self.sess.run([train, loss, rec_loss, self.cl_loss],
                                                   feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print('training:', epoch + 1, 'batch', n, 'total_loss:',l, 'rec_loss:', rec_l,'cl_loss',cl_l)
            self.U, self.V = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])
            self.ranking_performance(epoch)
        #self.U, self.V = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])
        self.U,self.V = self.bestU,self.bestV

    def predictForRanking(self, u):
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items
