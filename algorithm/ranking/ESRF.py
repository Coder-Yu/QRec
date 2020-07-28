# coding:utf8
from baseclass.DeepRecommender import DeepRecommender
from baseclass.SocialRecommender import SocialRecommender
from random import choice
import tensorflow as tf
from scipy.sparse import coo_matrix,spdiags
from math import sqrt
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def gumbel_softmax(logits, temperature=0.2):
    eps = 1e-10
    u = tf.random_uniform(tf.shape(logits), minval=0, maxval=1)
    gumbel_noise = -tf.log(-tf.log(u + eps) + eps)
    y = tf.log(logits + eps) + gumbel_noise
    return tf.nn.softmax(y / temperature)

class ESRF(SocialRecommender,DeepRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        DeepRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)
        self.beta = 0.2

    def buildSparseRelationMatrix(self):
        row, col, entries = [], [], []
        for pair in self.social.relation:
            # symmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.user[pair[1]]]
            entries += [1]
        AdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_users),dtype=np.float32)
        return AdjacencyMatrix

    def buildSparseRatingMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            # symmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.item[pair[1]]]
            entries += [1]
        ratingMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_items),dtype=np.float32)
        return ratingMatrix

    def buildMotifInducedAdjacencyMatrix(self):
        S = self.buildSparseRelationMatrix()
        Y = self.buildSparseRatingMatrix()
        self.userAdjacency = Y.tocsr()
        self.itemAdjacency = Y.transpose().tocsr()
        B = S.multiply(S.transpose())
        U = S - B
        # print 'S:', S.sum()
        # print 'B:', B.sum()
        # print 'U:', U.sum()

        Ds = spdiags(S.sum(axis=1).reshape(self.num_users),diags=[0],m=self.num_users,n=self.num_users)
        np.reciprocal(Ds.data, out=Ds.data,where=Ds.data!=0)
        S = Ds.dot(S)

        C1 = (U.dot(U)).multiply(U.transpose())
        A1 = C1 + C1.transpose()
        Da = spdiags(A1.sum(axis=1).reshape(self.num_users),diags=[0],m=self.num_users,n=self.num_users)
        np.reciprocal(Da.data,out=Da.data,where=Da.data!=0)
        # print Da.sum()
        # Da = np.sqrt(Da)
        A1 = Da.dot(A1)


        C2 = (B.dot(U)).multiply(U.transpose()) + (U.dot(B)).multiply(U.transpose()) + (U.dot(U)).multiply(B)
        A2 = C2 + C2.transpose()
        Da = spdiags(A2.sum(axis=1).reshape(self.num_users),diags=[0],m=self.num_users,n=self.num_users)
        np.reciprocal(Da.data,out=Da.data,where=Da.data!=0)
        A2 = Da.dot(A2)

        C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)).multiply(B)
        A3 = C3 + C3.transpose()
        Da = spdiags(A3.sum(axis=1).reshape(self.num_users),diags=[0],m=self.num_users,n=self.num_users)
        np.reciprocal(Da.data,out=Da.data,where=Da.data!=0)
        A3 = Da.dot(A3)

        A4 = (B.dot(B)).multiply(B)
        Da = spdiags(A4.sum(axis=1).reshape(self.num_users),diags=[0],m=self.num_users,n=self.num_users)
        np.reciprocal(Da.data,out=Da.data,where=Da.data!=0)
        A4 = Da.dot(A4)

        C5 = (U.dot(U)).multiply(U) + (U.dot(U.transpose())).multiply(U) + (U.transpose().dot(U)).multiply(U)
        A5 = C5 + C5.transpose()
        Da = spdiags(A5.sum(axis=1).reshape(self.num_users),diags=[0],m=self.num_users,n=self.num_users)
        np.reciprocal(Da.data,out=Da.data,where=Da.data!=0)
        A5 = Da.dot(A5)

        A6 = (U.dot(B)).multiply(U) + (B.dot(U.transpose())).multiply(U.transpose()) + (U.transpose().dot(U)).multiply(
            B)
        Da = spdiags(A6.sum(axis=1).reshape(self.num_users),diags=[0],m=self.num_users,n=self.num_users)
        np.reciprocal(Da.data,out=Da.data,where=Da.data!=0)
        A6 = Da.dot(A6)

        A7 = (U.transpose().dot(B)).multiply(U.transpose()) + (B.dot(U)).multiply(U) + (U.dot(U.transpose())).multiply(
            B)
        Da = spdiags(A7.sum(axis=1).reshape(self.num_users),diags=[0],m=self.num_users,n=self.num_users)
        np.reciprocal(Da.data,out=Da.data,where=Da.data!=0)
        A7 = Da.dot(A7)

        A8 = (Y.dot(Y.transpose())).multiply(U)
        Da = spdiags(A8.sum(axis=1).reshape(self.num_users),diags=[0],m=self.num_users,n=self.num_users)
        np.reciprocal(Da.data,out=Da.data,where=Da.data!=0)
        A8 = Da.dot(A8)

        A9 = (Y.dot(Y.transpose())).multiply(B)
        Da = spdiags(A9.sum(axis=1).reshape(self.num_users),diags=[0],m=self.num_users,n=self.num_users)
        np.reciprocal(Da.data,out=Da.data,where=Da.data!=0)
        A9 = Da.dot(A9)

        A10  = Y.dot(Y.transpose())
        mask = A10>=5
        A10 = A10.multiply(mask)
        Da = spdiags(A10.sum(axis=1).reshape(self.num_users),diags=[0],m=self.num_users,n=self.num_users)
        np.reciprocal(Da.data,out=Da.data,where=Da.data!=0)
        A10 = Da.dot(A10)

        A = S + A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 + A10
        #print A.todense()
        Da = spdiags(A.sum(axis=1).reshape(self.num_users),diags=[0],m=self.num_users,n=self.num_users)
        np.reciprocal(Da.data, out=Da.data,where=Da.data!=0)
        A = Da.dot(A)
        #print A.todense()
        return A

    def buildMotifGCN(self,adjacency):
        # self.isTraining = tf.placeholder(tf.int32)
        # self.isTraining = tf.cast(self.isTraining, tf.bool)
        self.relation_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embed_size], stddev=0.005),name='U_r')
        #convert sparse matrix to sparse tensor
        adjacency = adjacency.tocoo()
        indices = np.mat([adjacency.row, adjacency.col]).transpose()
        self.A = tf.SparseTensor(indices, adjacency.data.astype(np.float32), adjacency.shape)
        self.adjacency = adjacency.tocsr()
        self.g_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        self.n_layers_G = 3

        all_embeddings = [self.relation_embeddings]
        user_embeddings = self.relation_embeddings
        for d in range(self.n_layers_G):
            user_embeddings = tf.sparse_tensor_dense_matmul(self.A, user_embeddings)
            norm_embeddings = tf.math.l2_normalize(user_embeddings, axis=1)
            all_embeddings += [norm_embeddings]

        user_embeddings = tf.reduce_sum(all_embeddings, 0)

        # construct concrete selector layer
        self.K = 20
        # self.g_weights['c_selector'] = tf.Variable(
        #     initializer([self.n_layers_G * self.embed_size, self.K]), name='c_selector')
        self.g_weights['c_selector'] = tf.Variable(
            initializer([self.K,self.num_users]), name='c_selector')
        # def getAlternativeNeighborhood(embedding):
        # This piece of code often suffers from OOM error
        #     user_features = tf.multiply(embedding,user_embeddings)
        #     alphaEmbeddings = tf.transpose(tf.matmul(user_features, self.g_weights['c_selector']))
        #     one_hot_vector = tf.reduce_sum(self.sampling(alphaEmbeddings), 0)
        #     return one_hot_vector
        def getAlternativeNeighborhood(embedding):
            user_features = tf.matmul(tf.reshape(embedding,[1,self.embed_size]),user_embeddings,transpose_b=True)
            alphaEmbeddings = tf.multiply(user_features[0], self.g_weights['c_selector'])
            one_hot_vector = tf.reduce_sum(self.sampling(alphaEmbeddings), 0)
            return one_hot_vector

        self.alternativeNeighborhood = tf.vectorized_map(fn=lambda em:getAlternativeNeighborhood(em),elems=user_embeddings)


        #decoder
        reg_loss = 0
        decoder_weight_sizes = [self.num_users,self.embed_size*4,self.num_users]
        decoder_layers = 2
        for d in range(decoder_layers):
            self.g_weights['decoder_%d' % d] = tf.Variable(
                initializer([decoder_weight_sizes[d], decoder_weight_sizes[d + 1]]), name='decoder_%d' % d)
            reg_loss += tf.nn.l2_loss(self.g_weights['decoder_%d' % d])

        decoderEmbeddings = self.alternativeNeighborhood
        for d in range(decoder_layers-1):
            decoderEmbeddings = tf.matmul(decoderEmbeddings, self.g_weights['decoder_%d' % d])
            decoderEmbeddings = tf.nn.relu(decoderEmbeddings)
        decoderEmbeddings = tf.matmul(decoderEmbeddings, self.g_weights['decoder_%d' % (decoder_layers-1)])
        decoderEmbeddings = tf.nn.sigmoid(decoderEmbeddings)
        self.socialReconstruction = decoderEmbeddings
        #self.inputAdjacency = tf.placeholder(tf.float32, name="input_ad")
        self.mask = tf.placeholder(tf.float32, name="input_ad")

        #training
        self.r_loss = tf.nn.l2_loss(tf.multiply(self.mask,tf.sparse.add(-self.socialReconstruction,self.A)))+0.005*reg_loss
        r_opt = tf.train.AdamOptimizer(0.01)
        self.r_train = r_opt.minimize(self.r_loss)

    def buildRecGCN(self):
        self.isSocial = tf.placeholder(tf.int32)
        self.isSocial = tf.cast(self.isSocial, tf.bool)
        self.sampledItems = tf.placeholder(tf.int32)
        self.d_weights = dict()
        ego_embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)

        indices = [[self.data.user[item[0]], self.num_users + self.data.item[item[1]]] for item in
                   self.data.trainingData]
        indices += [[self.num_users + self.data.item[item[1]], self.data.user[item[0]]] for item in
                    self.data.trainingData]
        values = [float(item[2]) / sqrt(len(self.data.trainSet_u[item[0]])) / sqrt(len(self.data.trainSet_i[item[1]]))
                  for item in self.data.trainingData] * 2

        norm_adj = tf.SparseTensor(indices=indices, values=values,
                                   dense_shape=[self.num_users + self.num_items, self.num_users + self.num_items])

        indices = [[i,i] for i in range(self.num_users)]
        values = [1.0]*self.num_users
        assignMatrix = tf.SparseTensor(indices=indices, values=values,
                                   dense_shape=[self.num_users + self.num_items,self.num_users])
        initializer = tf.contrib.layers.xavier_initializer()

        self.n_layers_D = 4

        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers_D):
            self.d_weights['attention_m1%d' % k] = tf.Variable(
                initializer([self.embed_size,self.embed_size]), name='attention_m1%d' % k)
            self.d_weights['attention_m2%d' % k] = tf.Variable(
                initializer([self.embed_size, self.embed_size]), name='attention_m2%d' % k)
            self.d_weights['attention_m3%d' % k] = tf.Variable(
                initializer([self.embed_size, self.embed_size]), name='attention_m3%d' % k)
            self.d_weights['attention_v1%d' % k] = tf.Variable(
                initializer([1,self.embed_size]), name='attention_v1%d' % k)
            self.d_weights['attention_v2%d' % k] = tf.Variable(
                initializer([1,self.embed_size]), name='attention_v2%d' % k)
            self.d_weights['attention_v3%d' % k] = tf.Variable(
                initializer([1,self.embed_size]), name='attention_v3%d' % k)

        for k in range(self.n_layers_D):
            ego_embeddings = tf.sparse_tensor_dense_matmul(norm_adj, ego_embeddings)
            #social_embeddings = tf.matmul(self.alternativeNeighborhood, ego_embeddings[:self.num_users]) / self.K
            #ego_embeddings += tf.sparse_tensor_dense_matmul(assignMatrix, social_embeddings)

            # #social attention (applying attention may be a little time-consuming)
            # selectedItemEmbeddings = tf.gather(ego_embeddings[self.num_users:],self.sampledItems)
            # #hybridEmbeddings = tf.multiply(selectedItemEmbeddings,ego_embeddings[:self.num_users])
            # vals,indexes = tf.nn.top_k(self.alternativeNeighborhood,self.K)
            # indexes = tf.cast(indexes,tf.float32)
            # attentionEmbeddings = tf.concat([indexes,selectedItemEmbeddings],axis=1)
            # attentionEmbeddings = tf.concat([attentionEmbeddings,ego_embeddings[:self.num_users]],axis=1)
            #
            # def attention(embedding):
            #     alternativeNeighors,i_embedding,u_embedding = tf.split(tf.reshape(embedding,[1,self.K+2*self.embed_size]),[self.K,self.embed_size,self.embed_size],axis=1)
            #     alternativeNeighors = tf.cast(alternativeNeighors[0],tf.int32)
            #     # i_embedding = tf.matmul(i_embedding,self.d_weights['attention_m%d' % k])
            #     # u_embedding = tf.matmul(u_embedding, self.d_weights['attention_m%d' % k])
            #     friendsEmbedding = tf.gather(ego_embeddings[:self.num_users],alternativeNeighors)
            #     hybridEmbedding = tf.multiply(friendsEmbedding,i_embedding)
            #     hybridEmbedding = tf.multiply(hybridEmbedding,u_embedding)
            #     #weights 1:
            #     #weights1 = tf.matmul(hybridEmbedding, self.d_weights['attention_m1%d' % k])
            #     weights1 = tf.nn.relu(tf.matmul(self.d_weights['attention_v1%d' % k],hybridEmbedding,transpose_b=True))
            #     weights1 = tf.nn.softmax(weights1)
            #     # #weights 2:
            #     # weights2 = tf.matmul(hybridEmbedding, self.d_weights['attention_m2%d' % k])
            #     weights2 = tf.nn.relu(tf.matmul(self.d_weights['attention_v2%d' % k],hybridEmbedding,transpose_b=True))
            #     weights2 = tf.nn.softmax(weights2)
            #     # #weights 3:
            #     # weights3 = tf.matmul(hybridEmbedding, self.d_weights['attention_m3%d' % k])
            #     weights3 = tf.nn.relu(tf.matmul(self.d_weights['attention_v3%d' % k],hybridEmbedding,transpose_b=True))
            #     weights3 = tf.nn.softmax(weights3)
            #
            #     weights = (weights1+weights2+weights3)/3
            #     # weights = tf.matmul(self.d_weights['attention_v%d' % k],weights,transpose_b=True)
            #     # weights = tf.nn.softmax(weights)
            #     socialEmbedding = tf.matmul(tf.reshape(weights,[1,self.K]),tf.gather(ego_embeddings[:self.num_users],alternativeNeighors))
            #     return socialEmbedding[0]
            # socialEmbeddings = tf.vectorized_map(fn=lambda em: attention(em),elems=attentionEmbeddings)

            def without_social():
                return ego_embeddings
            def with_social(embeddings):
                socialEmbeddings = tf.matmul(self.alternativeNeighborhood, embeddings[:self.num_users])/self.K
                embeddings += tf.sparse_tensor_dense_matmul(assignMatrix, socialEmbeddings)
                return embeddings

            ego_embeddings = tf.cond(self.isSocial, lambda: with_social(ego_embeddings), lambda: without_social())
            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = tf.reduce_sum(all_embeddings, 0)
        self.multi_user_embeddings, self.multi_item_embeddings = tf.split(all_embeddings,[self.num_users, self.num_items], 0)
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_item_embedding = tf.nn.embedding_lookup(self.multi_item_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(self.multi_user_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.multi_item_embeddings, self.v_idx)

    def buildGenerator(self):
        y_ui = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1)
        currentNeighbors = tf.gather(self.alternativeNeighborhood, self.u_idx)
        friendEmbeddings = tf.matmul(currentNeighbors, self.multi_user_embeddings)/self.K
        #self.h_loss = tf.nn.l2_loss(self.u_embedding-self.friendEmbeddings)
        y_vi = tf.reduce_sum(tf.multiply(friendEmbeddings, self.v_embedding), 1)
        self.g_adv_loss = -tf.reduce_sum(tf.log(tf.sigmoid(y_vi-y_ui)))
        self.g_loss = 0.5*self.g_adv_loss+self.r_loss
        opt = tf.train.AdamOptimizer(self.lRate*10)
        self.g_train = opt.minimize(self.g_loss, var_list=[self.g_weights,self.relation_embeddings])

    def buildDiscriminator(self):
        #self.test = tf.reduce_sum(tf.multiply(self.u_embedding, self.multi_item_embeddings), 1)
        y_ui = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1)
        y_uj = tf.reduce_sum(tf.multiply(self.u_embedding, self.neg_item_embedding), 1)
        currentNeighbors = tf.gather(self.alternativeNeighborhood,self.u_idx)
        friendEmbeddings = tf.matmul(currentNeighbors, self.multi_user_embeddings) / self.K
        y_vi = tf.reduce_sum(tf.multiply(friendEmbeddings,self.v_embedding),1)
        s_Regularization = 0.03*tf.nn.l2_loss(self.u_embedding - friendEmbeddings)
        pairwise_loss = -tf.reduce_sum(tf.log(tf.sigmoid(y_ui-y_uj)))
        reg_loss = self.regU * (tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.v_embedding) + tf.nn.l2_loss(self.neg_item_embedding))
        adversarial_loss = -tf.reduce_sum(tf.log(tf.sigmoid(y_ui-y_vi)))
        self.d_loss = pairwise_loss + reg_loss
        self.d_advloss = pairwise_loss + reg_loss + 0.5*adversarial_loss+s_Regularization
        opt = tf.train.AdamOptimizer(self.lRate)
        self.d_train = opt.minimize(self.d_loss,var_list = [self.user_embeddings,self.item_embeddings])
        self.d_adv_train = opt.minimize(self.d_advloss, var_list=[self.user_embeddings, self.item_embeddings,self.d_weights])

    def sampling(self,vec):
        vec = tf.nn.softmax(vec)
        logits = gumbel_softmax(vec, 0.1)
        return logits

    def initModel(self):
        super(ESRF, self).initModel()
        self.listed_data = []
        for i in range(self.num_users):
            user = self.data.id2user[i]
            items = self.data.trainSet_u[user].keys()
            items = [self.data.item[item] for item in items]
            self.listed_data.append(items)

    def sampleItems(self):
        selectedItems = []
        for i in range(self.num_users):
            item = choice(self.listed_data[i])
            selectedItems.append(item)
        return selectedItems


    def buildModel(self):
        A = self.buildMotifInducedAdjacencyMatrix()
        self.buildMotifGCN(A)
        self.buildRecGCN()
        self.buildDiscriminator()
        self.buildGenerator()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        adjacency = self.adjacency.todense()
        mask = np.ones_like(adjacency)
        mask[adjacency>0]=10
        #pretrain Motif-based GCN
        for iteration in range(10):
            _, l = self.sess.run([self.r_train, self.r_loss],
                                 feed_dict={self.mask:mask})

            print 'training:', iteration + 1, 'loss:', l


        for iteration in range(self.maxIter/2):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx= batch
                selectedItems = self.sampleItems()
                _d,ld = self.sess.run([self.d_train,self.d_loss],
                                     feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx,self.isSocial:0,self.sampledItems:selectedItems})
                print 'D training_1:', iteration + 1, 'batch', n, 'D_loss:', ld

        for iteration in range(self.maxIter/2):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                selectedItems = self.sampleItems()
                _d, lad,ld,_g, lg = self.sess.run([self.d_adv_train, self.d_advloss, self.d_loss,self.g_train, self.g_adv_loss],
                                       feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx,self.isSocial:1,self.sampledItems:selectedItems,self.mask:mask})
                print 'D training:', iteration + 1, 'batch', n, 'D_Adv_loss:', lad, 'D_loss',ld
                print 'G training:', iteration + 1, 'batch', n, 'G_loss:', lg

            selectedItems = self.sampleItems()
            self.U, self.V = self.sess.run([self.multi_user_embeddings, self.multi_item_embeddings],
                                           feed_dict={self.u_idx: [0], self.neg_idx: [0],
                                                      self.v_idx: [0], self.isSocial: 1,self.sampledItems:selectedItems})
            self.isConverged(iteration + 1)
                #print 'H loss:', iteration + 1, 'batch', n, 'H_loss:', h

            # _, l = self.sess.run([self.r_train, self.r_loss],
            #                                    feed_dict={self.isTraining: 1, self.mask: mask})
            #print 'training:', iteration + 1, 'relation loss:', l
        selectedItems = self.sampleItems()
        self.U,self.V = self.sess.run([self.multi_user_embeddings,self.multi_item_embeddings],feed_dict={self.u_idx: [0], self.neg_idx: [0],
                                                                                                         self.v_idx: [0],self.isSocial:1,self.sampledItems:selectedItems})

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
            #return self.sess.run(self.test,feed_dict={self.u_idx:[u],self.isTraining:0})
        else:
            return [self.data.globalMean] * self.num_items
